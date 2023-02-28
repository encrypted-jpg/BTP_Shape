import torch
import torch.nn as nn
from extensions.chamfer_dist import ChamferDistanceL1


class FoldingVAE(nn.Module):
    def __init__(self, num_pred, encoder_channel=1024):
        super(FoldingVAE, self).__init__()
        self.num_pred = num_pred
        self.encoder_channel = encoder_channel
        self.grid_size = int(pow(self.num_pred, 0.5) + 0.5)

        self.first_conv = nn.Sequential(
            nn.Conv1d(3,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128,256,1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512,512,1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512,self.encoder_channel,1)
        )

        self.folding1 = nn.Sequential(
            nn.Conv1d(self.encoder_channel + 2, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 3, 1),
        )

        self.folding2 = nn.Sequential(
            nn.Conv1d(self.encoder_channel + 3, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 3, 1),
        )

        self.mu = nn.Linear(self.encoder_channel, self.encoder_channel)
        self.logvar = nn.Linear(self.encoder_channel, self.encoder_channel)

        a = torch.linspace(-0.5, 0.5, steps=self.grid_size, dtype=torch.float).view(1, self.grid_size).expand(self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.5, 0.5, steps=self.grid_size, dtype=torch.float).view(self.grid_size, 1).expand(self.grid_size, self.grid_size).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2).cuda() # 1 2 N
        self.chamfer = ChamferDistanceL1()

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encoder(self, xyz):
        xyz = xyz.transpose(2,1).contiguous()
        bs , n , _ = xyz.shape
        # print(xyz.shape)
        # encoder
        feature = self.first_conv(xyz.transpose(2,1))  # B 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # B 256 1
        # print(feature_global.expand(-1,-1,n).shape, feature.shape)
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# B 512 n
        feature = self.second_conv(feature) # B 1024 n
        feature_global = torch.max(feature,dim=2,keepdim=False)[0] # B 1024
        return feature_global
    
    def decoder(self,x):
        num_sample = self.grid_size * self.grid_size
        bs = x.size(0)
        features = x.view(bs, self.encoder_channel, 1).expand(bs, self.encoder_channel, num_sample)
        seed = self.folding_seed.view(1, 2, num_sample).expand(bs, 2, num_sample).to(x.device)

        x = torch.cat([seed, features], dim=1)
        fd1 = self.folding1(x)
        x = torch.cat([fd1, features], dim=1)
        fd2 = self.folding2(x)

        return fd1.transpose(2,1).contiguous() , fd2.transpose(2,1).contiguous()

    def forward(self, xyz):
        # encoder
        feature_global = self.encoder(xyz)
        # vae
        mu = self.mu(feature_global)
        logvar = self.logvar(feature_global)
        z = self.reparametrize(mu, logvar)
        # folding decoder
        fd1, fd2 = self.decoder(z) # B N 3
        return [fd1, fd2, mu, logvar]

    def loss_function(self, x, coarse, fine, mu, logvar, weight=[1, 1, 1]):
        coarseChamferLoss = self.chamfer(coarse, x)
        fineChamferLoss = self.chamfer(fine, x)
        kldLoss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        cw, fw, kw = weight
        return cw * coarseChamferLoss + fw * fineChamferLoss + kw * kldLoss, fineChamferLoss, kldLoss