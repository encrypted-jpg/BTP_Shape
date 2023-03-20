import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def farthest_point_sample(xyz, npoint):
    
    """
    code borrowed from: http://www.programmersought.com/article/8737853003/#14_query_ball_point_93
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
    	# Update the i-th farthest point
        centroids[:, i] = farthest
        # Take the xyz coordinate of the farthest point
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        # Calculate the Euclidean distance from all points in the point set to this farthest point
        dist = torch.sum((xyz - centroid) ** 2, -1)
        # Update distances to record the minimum distance of each point in the sample from all existing sample points
        mask = dist < distance
        distance[mask] = dist[mask]
        # Find the farthest point from the updated distances matrix, and use it as the farthest point for the next iteration
        farthest = torch.max(distance, -1)[1]
    return centroids

class kNNLoss(nn.Module):
    """
    Proposed PatchVariance component
    """
    def __init__(self, k=10, n_seeds=20):
        super(kNNLoss,self).__init__()
        self.k = k
        self.n_seeds = n_seeds
    def forward(self, pcs):
        n_seeds = self.n_seeds
        k = self.k
        seeds = farthest_point_sample(pcs,n_seeds) # which gives index
        seeds_value = torch.stack([pc[seed] for pc, seed in zip(pcs,seeds)]) 
        pcs_new = pcs.unsqueeze(2).repeat(1,1,n_seeds,1)
        seeds_new = seeds_value.unsqueeze(1).repeat(1,pcs_new.shape[1],1,1)
        dist = pcs_new.add(-seeds_new)
        dist_value = torch.norm(dist,dim=3)
        dist_new = dist_value.transpose(1,2)
        top_dist, idx = torch.topk(dist_new, k+1, dim=2, largest=False)
        overall_mean = top_dist[:,:,1:].mean()
        top_dist = top_dist/overall_mean
        var = torch.var(top_dist.mean(dim=2)).mean()
        return var

