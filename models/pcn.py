import torch
import torch.nn as nn
from extensions.chamfer_dist import ChamferDistanceL1


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

    def __init__(self, num_dense=16384, latent_dim=1024, grid_size=4):
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
        
        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2).cuda()  # (1, 2, S)

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
    def __init__(self, pcnModelPath, clusterPath, chamfer_dist, num_dense=6144, latent_dim=1024, grid_size=4):
        super().__init__()
        self.num_dense = num_dense
        self.latent_dim = latent_dim
        self.model = PCN(num_dense=num_dense, latent_dim=latent_dim, grid_size=grid_size)
        self.load_model(pcnModelPath)

        assert clusterPath != None, "[-] Cluster Path is None"

        self.kmeans, self.top5pcs = self.load_clusters(clusterPath)
        
        assert chamfer_dist != None, "[-] Chamfer Distance is None"
        
        self.chamfer_dist = chamfer_dist
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def forward(self, pc):
        return self.getNPC(pc)

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
        # for p in self.model.parameters():
        #     print(p.shape)
        # for x, y in checkpoint['model_state_dict'].items():
        #     print(x, y.shape)
        # for (p, (x, y)) in zip(self.model.parameters(), checkpoint["model_state_dict"].items()):
        #     print(x, p.shape, y.shape)
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
            dists.append(dist.item())
        return top5[np.argmin(dists)]

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
            nearestPC = self.getNearest(gt, top5)
            # gtidx = np.random.choice(nearestPC.shape[0], self.num_dense, replace=False)
            # nearestPC = nearestPC[gtidx]
            npcs.append(nearestPC)
        
        npcs = torch.from_numpy(np.array(npcs)).to(self.device)
        return npcs