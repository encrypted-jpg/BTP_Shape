import torch
import torch.nn as nn
from extensions.chamfer_dist import ChamferDistanceL1
import numpy as np
import pickle
import open3d as o3d
import copy
import json
import sys
sys.path.append("..")
from utils.pc_sample import farthest_point_sample, index_points


class PCN(nn.Module):
    """
    "PCN: Point Cloud Completion Network"
    (https://arxiv.org/pdf/1808.00671.pdf)
    Attributes:
        num_dense:  16384
        latent_dim: 1024
        grid_size:  4
        num_coarse: 1024
    """

    def __init__(self, num_dense=16384, latent_dim=1024, grid_size=4, device="cuda"):
        super().__init__()

        self.num_dense = num_dense
        self.latent_dim = latent_dim
        self.grid_size = grid_size

        assert self.num_dense % self.grid_size ** 2 == 0

        self.num_coarse = self.num_dense // (self.grid_size ** 2)

        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )

        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.latent_dim, 1)
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * self.num_coarse)
        )

        self.final_conv = nn.Sequential(
            nn.Conv1d(1024 + 3 + 2, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 3, 1)
        )
        a = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(1, self.grid_size).expand(self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(self.grid_size, 1).expand(self.grid_size, self.grid_size).reshape(1, -1)
        
        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2).to(device)  # (1, 2, S)

    def forward(self, xyz):
        B, N, _ = xyz.shape
        # encoder
        feature = self.first_conv(xyz.transpose(2, 1))                                       # (B,  256, N)
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]                          # (B,  256, 1)
        feature = torch.cat([feature_global.expand(-1, -1, N), feature], dim=1)              # (B,  512, N)
        feature = self.second_conv(feature)                                                  # (B, 1024, N)
        feature_global = torch.max(feature,dim=2,keepdim=False)[0]                           # (B, 1024)
        
        # decoder
        coarse = self.mlp(feature_global).reshape(-1, self.num_coarse, 3)                    # (B, num_coarse, 3), coarse point cloud
        point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size ** 2, -1)             # (B, num_coarse, S, 3)
        point_feat = point_feat.reshape(-1, self.num_dense, 3).transpose(2, 1)               # (B, 3, num_fine)

        seed = self.folding_seed.unsqueeze(2).expand(B, -1, self.num_coarse, -1)             # (B, 2, num_coarse, S)
        seed = seed.reshape(B, -1, self.num_dense)                                           # (B, 2, num_fine)

        feature_global = feature_global.unsqueeze(2).expand(-1, -1, self.num_dense)          # (B, 1024, num_fine)
        feat = torch.cat([feature_global, seed, point_feat], dim=1)                          # (B, 1024+2+3, num_fine)
    
        fine = self.final_conv(feat) + point_feat                                            # (B, 3, num_fine), fine point cloud

        return coarse.contiguous(), fine.transpose(1, 2).contiguous()
    
    def get_representation(self, xyz):
        B, N, _ = xyz.shape
        # encoder
        feature = self.first_conv(xyz.transpose(2, 1))                                       # (B,  256, N)
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]                          # (B,  256, 1)
        feature = torch.cat([feature_global.expand(-1, -1, N), feature], dim=1)              # (B,  512, N)
        feature = self.second_conv(feature)                                                  # (B, 1024, N)
        feature_global = torch.max(feature,dim=2,keepdim=False)[0]                           # (B, 1024)
        return feature_global
    
    def get_out(self, feat):
        B, _ = feat.shape
        coarse = self.mlp(feat).reshape(-1, self.num_coarse, 3)                    # (B, num_coarse, 3), coarse point cloud
        point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size ** 2, -1)             # (B, num_coarse, S, 3)
        point_feat = point_feat.reshape(-1, self.num_dense, 3).transpose(2, 1)               # (B, 3, num_fine)

        seed = self.folding_seed.unsqueeze(2).expand(B, -1, self.num_coarse, -1)             # (B, 2, num_coarse, S)
        seed = seed.reshape(B, -1, self.num_dense)                                           # (B, 2, num_fine)

        feature_global = feat.unsqueeze(2).expand(-1, -1, self.num_dense)          # (B, 1024, num_fine)
        feat = torch.cat([feature_global, seed, point_feat], dim=1)                          # (B, 1024+2+3, num_fine)
    
        fine = self.final_conv(feat) + point_feat                                            # (B, 3, num_fine), fine point cloud

        return coarse.contiguous(), fine.transpose(1, 2).contiguous()

class PCNEncoder(nn.Module):
    def __init__(self, latent_dim=1024):
        super(PCNEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )

        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.latent_dim, 1)
        )

    def forward(self, xyz):
        B, N, _ = xyz.shape
        # encoder
        feature = self.first_conv(xyz.transpose(2, 1))                                       # (B,  256, N)
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]                          # (B,  256, 1)
        feature = torch.cat([feature_global.expand(-1, -1, N), feature], dim=1)              # (B,  512, N)
        feature = self.second_conv(feature)                                                  # (B, 1024, N)
        feature_global = torch.max(feature,dim=2,keepdim=False)[0]                           # (B, 1024)
        return feature_global

class PCNDecoder(nn.Module):
    def __init__(self, latent_dim=1024, num_dense=6144, grid_size=4):
        super(PCNDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.num_dense = num_dense
        self.grid_size = grid_size
        self.num_coarse = self.num_dense // (self.grid_size ** 2)
        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3 * self.num_coarse)
        )

        self.final_conv = nn.Sequential(
            nn.Conv1d(self.latent_dim + 2 + 3, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 3, 1)
        )

        a = torch.linspace(-1, 1, self.grid_size)
        b = torch.linspace(-1, 1, self.grid_size)
        a, b = torch.meshgrid(a, b)
        a = a.reshape(-1)
        b = b.reshape(-1)
        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2).cuda()  # (1, 2, S)

    def forward(self, feat):
        B, _ = feat.shape
        # decoder
        coarse = self.mlp(feature_global).reshape(-1, self.num_coarse, 3)                    # (B, num_coarse, 3), coarse point cloud
        point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size ** 2, -1)             # (B, num_coarse, S, 3)
        point_feat = point_feat.reshape(-1, self.num_dense, 3).transpose(2, 1)               # (B, 3, num_fine)

        seed = self.folding_seed.unsqueeze(2).expand(B, -1, self.num_coarse, -1)             # (B, 2, num_coarse, S)
        seed = seed.reshape(B, -1, self.num_dense)                                           # (B, 2, num_fine)

        feature_global = feature_global.unsqueeze(2).expand(-1, -1, self.num_dense)          # (B, 1024, num_fine)
        feat = torch.cat([feature_global, seed, point_feat], dim=1)                          # (B, 1024+2+3, num_fine)
    
        fine = self.final_conv(feat) + point_feat                                            # (B, 3, num_fine), fine point cloud

        return coarse.contiguous(), fine.transpose(1, 2).contiguous()

class PCNAE(nn.Module):
    def __init__(self, latent_dim=1024, num_dense=6144, grid_size=4):
        super(PCNAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_dense = num_dense
        self.grid_size = grid_size
        self.encoder = PCNEncoder(latent_dim)
        self.decoder = PCNDecoder(latent_dim, num_dense, grid_size)

    def forward(self, xyz):
        feat = self.encoder(xyz)
        coarse, fine = self.decoder(feat)
        return coarse, fine

class Cluster(nn.Module):
    def __init__(self, pcnModelPath, clusterPath, chamfer_dist, num_dense=6144, latent_dim=1024, grid_size=4, device="cuda", jsonPath=None):
        super().__init__()
        self.num_dense = num_dense
        self.latent_dim = latent_dim
        self.model = PCN(num_dense=num_dense, latent_dim=latent_dim, grid_size=grid_size, device=device)
        self.load_model(pcnModelPath)

        assert clusterPath != None, "[-] Cluster Path is None"

        self.kmeans, self.top5pcs = self.load_clusters(clusterPath)
        
        assert chamfer_dist != None, "[-] Chamfer Distance is None"
        
        self.chamfer_dist = ChamferDistanceL1()
        self.jsonPath = jsonPath
        self.json = None
        if self.jsonPath is not None:
            with open(self.jsonPath) as f:
                self.json = json.load(f)
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        self.reg_failed_count = 0

    def forward(self, pc, idx=None):
        temp_found = False
        if self.json is not None and idx is not None:
            pclist = []
            err = False
            for k in range(pc.shape[0]):
                try:
                    key = idx[k].replace("\\", "/")
                    i, j = self.json[key]
                    pclist.append(np.array(self.top5pcs[i][j]))
                except KeyError:
                    err = True
                    print("[-] Error in JSON with Key: ", key)
            if not err:
                temp_found = True
                temp = np.array(pclist)
        with torch.no_grad():
            source_point_cloud = pc.detach().cpu().numpy()
            if not temp_found:
                template_point_cloud = self.getNPC(pc).detach().cpu().numpy()
            else:
                template_point_cloud = temp
            
            voxel_size = 5
            self.reg_failed_count = 0
            transformed_point_clouds = []
            npcs = []
            targets = []
            for i in range(source_point_cloud.shape[0]):
                source, target, source_down, target_down, source_fpfh, target_fpfh = self.prepare_dataset(voxel_size, source_point_cloud[i].reshape(-1, 3), template_point_cloud[i].reshape(-1, 3))
                result_ransac = self.execute_global_registration(source_down, target_down,
                                                            source_fpfh, target_fpfh,
                                                            voxel_size)
                result_icp = self.refine_registration(source, target, source_fpfh, target_fpfh,
                                                voxel_size, result_ransac, 0.002)
                original = copy.deepcopy(source)
                source.transform(result_icp.transformation)
                original = torch.from_numpy(np.array(original.points)).to(torch.float32).detach().requires_grad_(False)
                source = torch.from_numpy(np.array(source.points)).to(torch.float32).detach().requires_grad_(False)
                target = torch.from_numpy(np.array(target.points)).to(torch.float32).detach().requires_grad_(False)
                newPc = torch.from_numpy(np.concatenate((source, target), axis=0)).to(torch.float32).detach().requires_grad_(False)
                originalPc = torch.from_numpy(np.concatenate((original, target), axis=0)).to(torch.float32).detach().requires_grad_(False)
                targets.append(target)
                npcs.append(newPc)
                transformed_point_clouds.append(self.getBest(original.reshape(1, -1, 3), originalPc.reshape(1, -1, 3), newPc.reshape(1, -1, 3)))
                # transformed_point_clouds.append(newPc)
            transformed_point_cloud = torch.cat([transformed_point_cloud.reshape(1, -1, 3) for transformed_point_cloud in transformed_point_clouds], axis=0).to(self.device).to(torch.float32)
        nidx = farthest_point_sample(transformed_point_cloud, 1000, RAN = True)
        transformed_point_cloud = index_points(transformed_point_cloud, nidx)
        return transformed_point_cloud.to(torch.float32), template_point_cloud, targets, npcs
    
    def getBest(self, source, target, newPc):
        with torch.no_grad():
            dist1 = self.chamfer_dist(source.detach().cuda(device=0), target.detach().cuda(device=0))
            dist2 = self.chamfer_dist(source.detach().cuda(device=0), newPc.detach().cuda(device=0))
            if dist1.item() < dist2.item():
                self.reg_failed_count += 1
                return target
            else:
                # print("[+] Reg Success")
                return newPc

    def load_clusters(self, path):
        modelDict = pickle.load(open(path, "rb"))
        kmeans = modelDict["kmeans"]
        top5pcs = modelDict["top5pcs"]
        return kmeans, top5pcs

    def load_model(self, path):
        if path == None:
            return None
        print("[+] Loading Model from: {}".format(path))
        checkpoint = torch.load(path)
        try:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print("[+] Model Statistics - Epoch: {}, Loss: {}".format(checkpoint["epoch"], checkpoint["loss"]))
        except:
            self.model.load_state_dict(checkpoint)
    
    def getNearest(self, pc, top5):
        dists = []
        for i in range(len(top5)):
            cmp = torch.Tensor(np.array([top5[i]])).to(torch.float32).to(self.device)
            dist = self.chamfer_dist(pc, cmp)
            dists.append(dist.detach().cpu().numpy())
        return top5[np.argmin(dists)], np.argmin(dists)

    def getNPC(self, pc):
        # gt = torch.Tensor(np.array([pc])).to(torch.float32).to(self.device)
        gt = pc.detach().to(torch.float32)
        rep = self.model.get_representation(gt).detach().cpu().numpy()
        # for i, val in enumerate(rep):
        #     if val.any() == np.NaN:
        #         rep[i] = np.zeros(self.latent_dim)
        labels = self.kmeans.predict(rep)
        npcs = []
        for label in labels.tolist():
            top5 = self.top5pcs[label]
            nearestPC, _ = self.getNearest(gt, top5)
            # gtidx = np.random.choice(nearestPC.shape[0], self.num_dense, replace=False)
            # nearestPC = nearestPC[gtidx]
            npcs.append(nearestPC)
        
        npcs = torch.from_numpy(np.array(npcs)).to(self.device)
        return npcs
    
    def getIdx(self, pc):
        # gt = torch.Tensor(np.array([pc])).to(torch.float32).to(self.device)
        gt = pc.detach().to(torch.float32)
        rep = self.model.get_representation(gt).detach().cpu().numpy()
        # for i, val in enumerate(rep):
        #     if val.any() == np.NaN:
        #         rep[i] = np.zeros(self.latent_dim)
        labels = self.kmeans.predict(rep)
        npcs = []
        for label in labels.tolist():
            top5 = self.top5pcs[label]
            _, idx = self.getNearest(gt, top5)
            # gtidx = np.random.choice(nearestPC.shape[0], self.num_dense, replace=False)
            # nearestPC = nearestPC[gtidx]
            npcs.append((int(label), int(idx)))
        
        return npcs
    
    def draw_registration_result(self, source, target, window_name="Result"):
        """
        Displays registration result
        :param window_name: name of window
        :param source: source PointCloud
        :param target: target PointCloud
        :param transformation: transformation from target to source
        :return:
        """
        source = copy.deepcopy(source)
        target = copy.deepcopy(target)
        source_temp = o3d.geometry.PointCloud()
        source_temp.points = o3d.utility.Vector3dVector(source)
        target_temp = o3d.geometry.PointCloud()
        target_temp.points = o3d.utility.Vector3dVector(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        # source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp], window_name=window_name)

    def preprocess_point_cloud(self, pcd, voxel_size):
        """
        Resamples point cloud and computes normals
        :param pcd: point cloud
        :param voxel_size: size of voxel
        :return: resampled pcd and features
        """
        # if LOG:
        #     print("INFO: Downsampling with a voxel size %.3f." % voxel_size)
        pcd_down = pcd.voxel_down_sample(voxel_size)

        radius_normal = voxel_size * 2
        # if LOG:
        #     print("INFO: Estimating normals with search radius %.3f." % radius_normal)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size * 5
        # if LOG:
        #     print("INFO: Computing FPFH features with search radius %.3f." % radius_feature)
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh

    def prepare_dataset(self, voxel_size, source, target):
        """
        Loads and prepares dataset
        :param voxel_size: size of voxel to resample
        :param file1:
        :param file2:
        :return: pcds, resampled pcds, features
        """
        # if LOG:
        #     print("INFO: Load two point clouds.")
        # "./data/data10_points.ply"
        # "./data/headFace3_geo_low.ply"
        # source = o3d.io.read_point_cloud(file1)
        # target = o3d.io.read_point_cloud(file2)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(source)
        source = pcd
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(target)
        target = pcd

        source_down, source_fpfh = self.preprocess_point_cloud(source, voxel_size)
        target_down, target_fpfh = self.preprocess_point_cloud(target, voxel_size)
        return source, target, source_down, target_down, source_fpfh, target_fpfh

    def execute_global_registration(self, source_down, target_down, source_fpfh, target_fpfh, voxel_size):
        """
        Excecutes global registration using RANSAC
        :param source_down: resampled pcd
        :param target_down: resampled pcd
        :param source_fpfh: features
        :param target_fpfh: features
        :param voxel_size: size of voxel
        :return:
        """
        distance_threshold = voxel_size * 0.5
        # if LOG:
        #     print("INFO: Launching global registration using RANSAC")
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
        return result

    def refine_registration(self, source, target, source_fpfh, target_fpfh, voxel_size, result_ransac, distance_threshold=None):
        if not distance_threshold:
            distance_threshold = voxel_size * 0.3
        # if LOG:
        #     print("INFO: Running point-to-plane ICP registration")
        result = o3d.pipelines.registration.registration_icp(
            source, target, distance_threshold, result_ransac.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        return result



