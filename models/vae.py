import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import open3d as o3d
import itertools
from extensions.chamfer_dist import ChamferDistanceL1

class Encoder(nn.Module):
    def __init__(self, global_feat=False, channel=3):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv1d(channel, 128, 1)
        # self.bn1 = nn.BatchNorm1d(128)  no bn in PCN
        self.conv2 = nn.Conv1d(128, 256, 1)
        self.conv3 = nn.Conv1d(512, 512, 1)
        self.conv4 = nn.Conv1d(512, 1024, 1)
        self.global_feat = global_feat

    def forward(self, x):
        _, D, N = x.size()
        x = F.relu(self.conv1(x))
        pointfeat = self.conv2(x)

        # 'encoder_0'
        feat = torch.max(pointfeat, 2, keepdim=True)[0]
        feat = feat.view(-1, 256, 1).repeat(1, 1, N)
        x = torch.cat([pointfeat, feat], 1)

        # 'encoder_1'
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        x = torch.max(x, 2, keepdim=False)[0]

        if self.global_feat:  # used in completion and classification tasks
            return x
        else:  # concatenate global and local features, for segmentation tasks
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1)


class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__()

        self.grid_size = 4
        self.grid_scale = 0.05
        self.num_coarse = 403
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.__dict__.update(kwargs)  # to update args, num_coarse, grid_size, grid_scale

        self.num_fine = self.grid_size ** 2 * self.num_coarse  # 16384
        self.meshgrid = [[-self.grid_scale, self.grid_scale, self.grid_size],
                         [-self.grid_scale, self.grid_scale, self.grid_size]]

        # batch normalisation will destroy limit the expression
        self.folding1 = nn.Sequential(
            nn.Linear(1024, 1024),
            # nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            # nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, self.num_coarse * 3))

        self.folding2 = nn.Sequential(
            nn.Conv1d(1024+2+3, 512, 1),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 3, 1))

    def build_grid(self, batch_size):
        # a simpler alternative would be: torch.meshgrid()
        x, y = np.linspace(*self.meshgrid[0]), np.linspace(*self.meshgrid[1])
        points = np.array(list(itertools.product(x, y)))
        points = np.repeat(points[np.newaxis, ...], repeats=batch_size, axis=0)

        return torch.tensor(points).float().to(self.device)

    def tile(self, tensor, multiples):
        # substitute for tf.tile:
        # https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/tile
        # Ref: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        def tile_single_axis(a, dim, n_tile):
            init_dim = a.size()[dim]
            repeat_idx = [1] * a.dim()
            repeat_idx[dim] = n_tile
            a = a.repeat(*repeat_idx)
            order_index = torch.Tensor(
                np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).long()
            return torch.index_select(a, dim, order_index.to(self.device))

        for dim, n_tile in enumerate(multiples):
            if n_tile == 1:  # increase the speed effectively
                continue
            tensor = tile_single_axis(tensor, dim, n_tile)
        return tensor

    @staticmethod
    def expand_dims(tensor, dim):
        # substitute for tf.expand_dims:
        # https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/expand_dims
        # another solution is: torch.unsqueeze(tensor, dim=dim)
        return tensor.unsqueeze(-1).transpose(-1, dim)

    def forward(self, feature):
        # use the same variable naming as:
        # https://github.com/wentaoyuan/pcn/blob/master/models/pcn_cd.py

        coarse = self.folding1(feature)
        coarse = coarse.view(-1, self.num_coarse, 3)

        grid = self.build_grid(feature.shape[0])
        grid_feat = grid.repeat(1, self.num_coarse, 1)

        point_feat = self.tile(self.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
        point_feat = point_feat.view([-1, self.num_fine, 3])

        global_feat = self.tile(self.expand_dims(feature, 1), [1, self.num_fine, 1])
        feat = torch.cat([grid_feat, point_feat, global_feat], dim=2)

        center = self.tile(self.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
        center = center.view([-1, self.num_fine, 3])

        fine = self.folding2(feat.transpose(2, 1)).transpose(2, 1) + center

        return coarse, fine

class VAE(nn.Module):
    """
    This class implements Variational Auto-Encoder.
    Write separate functions for encoding and decoding.
    It takes input of a 3d point cloud.
    It has an intermediate representation in latent space.
    It outputs a 3d Point Cloud.
    """
    def __init__(self, latent_dim=1024):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)
        self.encoder = Encoder(global_feat=True, channel=3)
        self.decoder = Decoder()
        self.chamfer = ChamferDistanceL1()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        eout = self.encoder(x)
        mu = self.fc_mu(eout)
        logvar = self.fc_logvar(eout)
        z = self.reparameterize(mu, logvar)
        coarse, fine = self.decoder(z)
        return [coarse, fine, mu, logvar]
    
    def loss_function(self, x, coarse, fine, mu, logvar):
        coarseChamferLoss = self.chamfer(coarse, x)
        fineChamferLoss = self.chamfer(fine, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return coarseChamferLoss + fineChamferLoss + KLD
    
    def generate(self, x):
        return self.forward(x)[1]
    







    
