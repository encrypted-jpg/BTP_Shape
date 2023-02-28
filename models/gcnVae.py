import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from extensions.chamfer_dist import ChamferDistanceL1

def get_neighbor_index(vertices, neighbor_num: int):
    """
    Return: (bs, vertice_num, neighbor_num)
    """
    bs, v, _ = vertices.size()
    device = vertices.device
    inner = torch.bmm(vertices, vertices.transpose(1, 2)) #(bs, v, v)
    quadratic = torch.sum(vertices**2, dim= 2) #(bs, v)
    distance = inner * (-2) + quadratic.unsqueeze(1) + quadratic.unsqueeze(2)
    k = min(neighbor_num + 1, v)
    neighbor_index = torch.topk(distance, k=k, dim= -1, largest= False)[1]
    neighbor_index = neighbor_index[:, :, 1:]
    return neighbor_index

def get_nearest_index(target, source):
    """
    Return: (bs, v1, 1)
    """
    inner = torch.bmm(target, source.transpose(1, 2)) #(bs, v1, v2)
    s_norm_2 = torch.sum(source ** 2, dim= 2) #(bs, v2)
    t_norm_2 = torch.sum(target ** 2, dim= 2) #(bs, v1)
    d_norm_2 = s_norm_2.unsqueeze(1) + t_norm_2.unsqueeze(2) - 2 * inner
    nearest_index = torch.topk(d_norm_2, k= 1, dim= -1, largest= False)[1]
    return nearest_index

def indexing_neighbor(tensor, index):
    """
    Return: (bs, vertice_num, neighbor_num, dim)
    """
    bs, v, n = index.size()
    id_0 = torch.arange(bs).view(-1, 1, 1)
    tensor_indexed = tensor[id_0, index]
    return tensor_indexed

def get_neighbor_direction_norm(vertices, neighbor_index):
    """
    Return: (bs, vertice_num, neighobr_num, 3)
    """
    neighbors = indexing_neighbor(vertices, neighbor_index) # (bs, v, n, 3)
    neighbor_direction = neighbors - vertices.unsqueeze(2)
    neighbor_direction_norm = F.normalize(neighbor_direction, dim= -1)
    return neighbor_direction_norm

class Conv_surface(nn.Module):
    """Extract structure feafure from surface, independent from vertice coordinates"""
    def __init__(self, kernel_num, support_num):
        super().__init__()
        self.kernel_num = kernel_num
        self.support_num = support_num

        self.relu = nn.ReLU(inplace= True)
        self.directions = nn.Parameter(torch.FloatTensor(3, support_num * kernel_num))
        self.initialize()

    def initialize(self):
        stdv = 1. / math.sqrt(self.support_num * self.kernel_num)
        self.directions.data.uniform_(-stdv, stdv)
    
    def forward(self, neighbor_index, vertices):
        """
        Return vertices with local feature: (bs, vertice_num, kernel_num)
        """
        bs, vertice_num, neighbor_num = neighbor_index.size() 
        neighbor_direction_norm = get_neighbor_direction_norm(vertices, neighbor_index)
        support_direction_norm = F.normalize(self.directions, dim= 0) #(3, s * k)
        neighbor_direction_norm = neighbor_direction_norm.double()
        support_direction_norm = support_direction_norm.double()
        print(neighbor_direction_norm.shape, support_direction_norm.shape)
        theta = neighbor_direction_norm @ support_direction_norm # (bs, vertice_num, neighbor_num, s*k)

        theta = self.relu(theta)
        theta = theta.contiguous().view(bs, vertice_num, neighbor_num, self.support_num, self.kernel_num)
        theta = torch.max(theta, dim= 2)[0] # (bs, vertice_num, support_num, kernel_num)
        feature = torch.sum(theta, dim= 2) # (bs, vertice_num, kernel_num)
        return feature

class Conv_layer(nn.Module):
    def __init__(self, in_channel, out_channel, support_num):
        super().__init__()
        # arguments: 
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.support_num = support_num

        # parameters:
        self.relu = nn.ReLU(inplace= True)
        self.weights = nn.Parameter(torch.FloatTensor(in_channel, (support_num + 1) * out_channel))
        self.bias = nn.Parameter(torch.FloatTensor((support_num + 1) * out_channel))
        self.directions = nn.Parameter(torch.FloatTensor(3, support_num * out_channel))
        self.initialize()

    def initialize(self):
        stdv = 1. / math.sqrt(self.out_channel * (self.support_num + 1))
        self.weights.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
        self.directions.data.uniform_(-stdv, stdv)

    def forward(self, neighbor_index, vertices, feature_map):
        """
        Return: output feature map: (bs, vertice_num, out_channel)
        """
        bs, vertice_num, neighbor_num = neighbor_index.size()
        neighbor_direction_norm = get_neighbor_direction_norm(vertices, neighbor_index)
        support_direction_norm = F.normalize(self.directions, dim= 0)
        neighbor_direction_norm = neighbor_direction_norm.double()
        support_direction_norm = support_direction_norm.double()
        theta = neighbor_direction_norm @ support_direction_norm # (bs, vertice_num, neighbor_num, support_num * out_channel)
        theta = self.relu(theta)
        theta = theta.contiguous().view(bs, vertice_num, neighbor_num, -1)
        # (bs, vertice_num, neighbor_num, support_num * out_channel)

        feature_map = feature_map.float()
        feature_out = feature_map @ self.weights + self.bias # (bs, vertice_num, (support_num + 1) * out_channel)
        feature_center = feature_out[:, :, :self.out_channel] # (bs, vertice_num, out_channel)
        feature_support = feature_out[:, :, self.out_channel:] #(bs, vertice_num, support_num * out_channel)

        # Fuse together - max among product
        feature_support = indexing_neighbor(feature_support, neighbor_index) # (bs, vertice_num, neighbor_num, support_num * out_channel)
        activation_support = theta * feature_support # (bs, vertice_num, neighbor_num, support_num * out_channel)
        activation_support = activation_support.view(bs,vertice_num, neighbor_num, self.support_num, self.out_channel)
        activation_support = torch.max(activation_support, dim= 2)[0] # (bs, vertice_num, support_num, out_channel)
        activation_support = torch.sum(activation_support, dim= 2)    # (bs, vertice_num, out_channel)
        feature_fuse = feature_center + activation_support # (bs, vertice_num, out_channel)
        return feature_fuse

class Pool_layer(nn.Module):
    def __init__(self, pooling_rate: int= 4, neighbor_num: int=  4):
        super().__init__()
        self.pooling_rate = pooling_rate
        self.neighbor_num = neighbor_num

    def forward(self, vertices, feature_map):
        """
        Return:
            vertices_pool: (bs, pool_vertice_num, 3),
            feature_map_pool: (bs, pool_vertice_num, channel_num)
        """
        bs, vertice_num, _ = vertices.size()
        neighbor_index = get_neighbor_index(vertices, self.neighbor_num)
        neighbor_feature = indexing_neighbor(feature_map, neighbor_index) #(bs, vertice_num, neighbor_num, channel_num)
        pooled_feature = torch.max(neighbor_feature, dim= 2)[0] #(bs, vertice_num, channel_num)

        pool_num = int(vertice_num / self.pooling_rate)
        sample_idx = torch.randperm(vertice_num)[:pool_num]
        vertices_pool = vertices[:, sample_idx, :] # (bs, pool_num, 3)
        feature_map_pool = pooled_feature[:, sample_idx, :] #(bs, pool_num, channel_num)
        return vertices_pool, feature_map_pool

class Conv_transpose_layer(nn.Module):
    def __init__(self, in_channel, out_channel, support_num):
        super().__init__()
        # arguments: 
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.support_num = support_num

        # parameters:
        self.relu = nn.ReLU(inplace= True)
        self.weights = nn.Parameter(torch.FloatTensor(in_channel, (support_num + 1) * out_channel))
        self.bias = nn.Parameter(torch.FloatTensor(in_channel))
        self.directions = nn.Parameter(torch.FloatTensor(3, support_num * out_channel))
        self.initialize()

    def initialize(self):
        stdv = 1. / math.sqrt(self.out_channel * (self.support_num + 1))
        self.weights.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
        self.directions.data.uniform_(-stdv, stdv)

    def forward(self, neighbor_index, vertices, feature_map):
        """
        Return: output feature map: (bs, vertice_num, out_channel)
        """
        bs, vertice_num, neighbor_num = neighbor_index.size()
        neighbor_direction_norm = get_neighbor_direction_norm(vertices, neighbor_index)
        support_direction_norm = F.normalize(self.directions, dim= 0) #(3, s * k)
        theta = neighbor_direction_norm @ support_direction_norm # (bs, vertice_num, neighbor_num, s*k)

        theta = self.relu(theta)
        theta = theta.contiguous().view(bs, vertice_num, neighbor_num, self.support_num, self.out_channel)
        theta = torch.max(theta, dim= 2)[0] # (bs, vertice_num, support_num, out_channel)
        feature_map = feature_map.unsqueeze(2) # (bs, vertice_num, 1, in_channel)
        feature_map = feature_map.double()
        theta = theta.double()
        print(feature_map.shape, theta.shape, theta.transpose(2, 3).shape)
        output = feature_map @ theta.transpose(2, 3) # (bs, vertice_num, 1, out_channel)
        output = output.squeeze(2) # (bs, vertice_num, out_channel)
        output = output + self.bias.unsqueeze(0) # (bs, vertice_num, out_channel)
        print(output.shape)
        return output

class GCN3DEncoder(nn.Module):
    def __init__(self, support_num, neighbor_num):
        super().__init__()
        self.neighbor_num = neighbor_num

        self.conv_0 = Conv_surface(kernel_num= 32, support_num= support_num)
        self.conv_1 = Conv_layer(32, 64, support_num= support_num)
        self.pool_1 = Pool_layer(pooling_rate= 4, neighbor_num= 4)
        self.conv_2 = Conv_layer(64, 128, support_num= support_num)
        self.conv_3 = Conv_layer(128, 256, support_num= support_num)
        self.pool_2 = Pool_layer(pooling_rate= 4, neighbor_num= 4)
        self.conv_4 = Conv_layer(256, 1024, support_num= support_num)
        self.linear_1 = nn.Linear(1024, 512)

    def forward(self,  vertices):
        bs, vertice_num, _ = vertices.size()
        
        neighbor_index = get_neighbor_index(vertices, self.neighbor_num)
        fm_0 = self.conv_0(neighbor_index, vertices)
        fm_0 = F.relu(fm_0, inplace= True)
        fm_1 = self.conv_1(neighbor_index, vertices, fm_0)
        fm_1 = F.relu(fm_1, inplace= True)
        vertices, fm_1 = self.pool_1(vertices, fm_1)
        neighbor_index = get_neighbor_index(vertices, self.neighbor_num)

        fm_2 = self.conv_2(neighbor_index, vertices, fm_1)
        fm_2 = F.relu(fm_2, inplace= True) 
        fm_3 = self.conv_3(neighbor_index, vertices, fm_2)
        fm_3 = F.relu(fm_3, inplace= True) 
        vertices, fm_3 = self.pool_2(vertices, fm_3)
        neighbor_index = get_neighbor_index(vertices, self.neighbor_num)
        
        fm_4 = self.conv_4(neighbor_index, vertices, fm_3)
        feature_global = fm_4.max(1)[0]
        feature_global = feature_global.to(torch.float32)
        feature_global = self.linear_1(feature_global)
        return feature_global

class GCN3DDecoder(nn.Module):
    def __init__(self, support_num, neighbor_num):
        super().__init__()
        self.neighbor_num = neighbor_num

        self.linear_1 = nn.Linear(512, 1024)
        self.conv_1 = Conv_surface(kernel_num= 32, support_num= support_num)
        self.conv_trans_1 = Conv_transpose_layer(1024, 32, support_num= support_num)
        self.conv_2 = Conv_layer(256, 128, support_num= support_num)
        self.conv_trans_2 = Conv_transpose_layer(32, 16, support_num= support_num)
        self.conv_3 = Conv_layer(64, 32, support_num= support_num)
        self.conv_trans_3 = Conv_transpose_layer(16, 3, support_num= support_num)

    def forward(self, feature_global):
        bs = feature_global.size(0)
        
        fm_0 = self.linear_1(feature_global)
        fm_0 = fm_0.view(bs, -1, 1)
        fm_0 = fm_0.repeat(1, 1, self.neighbor_num)
        fm_0 = fm_0.view(bs, -1, 1024)

        neighbor_index = get_neighbor_index(fm_0, self.neighbor_num)
        fm_1 = self.conv_1(neighbor_index, fm_0)
        fm_1 = F.relu(fm_1, inplace= True)
        print(neighbor_index.size())
        fm_2 = self.conv_trans_1(neighbor_index, fm_0, fm_1)
        fm_2 = F.relu(fm_2, inplace= True)
        # fm_3 = self.conv_2(neighbor_index, fm_0, fm_2)
        # fm_3 = F.relu(fm_3, inplace= True)
        fm_4 = self.conv_trans_2(neighbor_index, fm_0, fm_2)
        fm_4 = F.relu(fm_4, inplace= True)
        # fm_5 = self.conv_3(neighbor_index, fm_0, fm_4)
        # fm_5 = F.relu(fm_5, inplace= True)
        fm_6 = self.conv_trans_3(neighbor_index, fm_0, fm_4)
        fm_6 = torch.tanh(fm_6)

        return fm_6


class GCN3DVAE(nn.Module):
    def __init__(self, support_num, neighbor_num):
        super().__init__()
        self.encoder = GCN3DEncoder(support_num, neighbor_num)
        self.decoder = GCN3DDecoder(support_num, neighbor_num)
        self.mu = nn.Linear(512, 512)
        self.logvar = nn.Linear(512, 512)
        self.chamfer = ChamferDistanceL1()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, vertices):
        feature_global = self.encoder(vertices)
        mu = self.mu(feature_global)
        logvar = self.logvar(feature_global)
        feature_log = self.reparameterize(mu, logvar)
        vertices = self.decoder(feature_log)
        return [vertices, mu, logvar]
    
    def loss_function(self, x, fine, mu, logvar, weight=[1, 1]):
        fineChamferLoss = self.chamfer(fine, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        fw, kw = weight
        return fw * fineChamferLoss + kw * KLD, fineChamferLoss, KLD
