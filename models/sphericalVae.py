import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from extensions.chamfer_dist import ChamferDistanceL1

class SphericalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SphericalConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x, adj):
        # x: input point cloud of size (batch_size, num_points, in_channels)
        # adj: adjacency matrix of size (batch_size, num_points, k)
        batch_size, num_points, in_channels = x.size()
        k = adj.size(-1)

        # Compute local frames for each point
        knn_idx = adj[:,:,1:].contiguous() # exclude self-connection
        knn = x.view(batch_size, num_points, 1, in_channels).repeat(1, 1, k-1, 1)
        knn = torch.gather(x, 1, knn_idx.unsqueeze(-1).repeat(1, 1, 1, in_channels))
        diff = knn - x.view(batch_size, num_points, 1, in_channels)
        norm = torch.norm(diff, p=2, dim=-1, keepdim=True)
        frame = torch.cat((diff, norm), dim=-1)
        frame = frame / torch.norm(frame, p=2, dim=-1, keepdim=True)

        # Project points onto unit sphere
        radius = 1.0
        proj = radius * frame[:,:,:-1]

        # Compute convolutional weights
        weights = self.weight.view(1, self.out_channels, self.in_channels, self.kernel_size)\
            .repeat(batch_size, 1, 1, 1)

        # Compute dot product between kernel weights and local frames
        weights = weights.view(batch_size*self.out_channels, self.in_channels*self.kernel_size)
        proj = proj.view(batch_size, num_points*(k-1), self.in_channels)
        proj = proj.transpose(1, 2).contiguous().view(batch_size*self.in_channels, num_points*(k-1))
        output = torch.mm(weights, proj).view(batch_size, self.out_channels, num_points)

        # Apply max pooling over k nearest neighbors
        output = F.relu(output)
        output = F.max_pool1d(output, k-1)
        return output


class SphericalConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(SphericalConvTranspose, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x, adj, output_size):
        # x: input feature map of size (batch_size, in_channels, num_points)
        # adj: adjacency matrix of size (batch_size, num_points, k)
        # output_size: size of output feature map (num_points, out_channels)
        batch_size, in_channels, num_points = x.size()
        k = adj.size(-1)
        output_num_points = output_size[0]
        output_channels = output_size[1]

        # Compute local frames for each point
        knn_idx = adj[:,:,1:].contiguous() # exclude self-connection
        knn = x.view(batch_size, in_channels, num_points, 1).repeat(1, 1, 1, k-1)
        knn = torch.gather(x, 2, knn_idx.unsqueeze(1).repeat(1, in_channels, 1, 1))
        diff = knn - x.view(batch_size, in_channels, num_points, 1)
        norm = torch.norm(diff, p=2, dim=1, keepdim=True)
        frame = torch.cat((diff, norm), dim=1)
        frame = frame / torch.norm(frame, p=2, dim=1, keepdim=True)

        # Project points onto unit sphere
        radius = 1.0
        proj = radius * frame[:,:,:-1]

        # Compute convolutional weights
        weights = self.weight.view(1, self.out_channels, self.in_channels, self.kernel_size)\
            .repeat(batch_size, 1, 1, 1)

        # Compute dot product between kernel weights and local frames
        weights = weights.unsqueeze(2).repeat(1, 1, k-1, 1, 1)
        frame = frame.unsqueeze(1).repeat(1, self.out_channels, 1, 1, 1)
        dot = torch.sum(weights * frame, dim=-1)
        dot = dot.view(batch_size, self.out_channels, num_points, k-1)

        # Perform transposed convolution
        output = torch.zeros(batch_size, self.out_channels, output_num_points, device=x.device)
        for i in range(k-1):
            j = knn_idx[:,:,i]
            output[:,:,j] += dot[:,:,:,i]
        output /= k

        return output
    

class SphericalVAE(nn.Module):
    def __init__(self, input_channels=3, hidden_size=128, latent_size=64):
        super(SphericalVAE, self).__init__()
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        self.encoder = nn.Sequential(
            SphericalConv(self.input_channels, self.hidden_size, kernel_size=3),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            SphericalConv(self.hidden_size, self.hidden_size, kernel_size=3),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            SphericalConv(self.hidden_size, self.hidden_size, kernel_size=3),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.hidden_size*3, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 2*self.latent_size)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, self.hidden_size*3),
            nn.BatchNorm1d(self.hidden_size*3),
            nn.ReLU(),
            nn.Unflatten(1, (self.hidden_size, 4096)),
            SphericalConvTranspose(self.hidden_size, self.hidden_size, kernel_size=3),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            SphericalConvTranspose(self.hidden_size, self.hidden_size, kernel_size=3),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            SphericalConvTranspose(self.hidden_size, self.input_channels, kernel_size=3)
        )
        self.chamfer = ChamferDistanceL1()

    def encode(self, x):
        h = self.encoder(x)
        mu, log_var = torch.split(h, self.latent_size, dim=1)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        z = mu + eps*std
        return z

    def decode(self, z):
        x_recon = self.decoder(z)
        return x_recon

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var
    
    def loss_function(self, x, fine, mu, logvar):
        fineChamferLoss = self.chamfer(fine, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return fineChamferLoss + KLD, fineChamferLoss, KLD
