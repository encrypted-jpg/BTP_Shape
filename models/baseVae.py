import torch
import torch.nn as nn
import torch.nn.functional as F
from extensions.chamfer_dist import ChamferDistanceL1

def sample_gaussian(m, v):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epsilon = torch.normal(torch.zeros(m.size()),torch.ones(m.size())).to(device)
    z = m + torch.sqrt(v) * epsilon
    return z

def kl_normal(qm,qv,pm,pv):
    # tensor shape (Batch,dim)
    element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
    kl = element_wise.sum(-1)
    return kl

class Encoder(nn.Module):
    def __init__(self, zdim, input_dim=3):
        super(Encoder, self).__init__()
        self.zdim = zdim
        self.conv1 = nn.Conv1d(input_dim, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        # Mapping to [c], cmean
        self.fc1_m = nn.Linear(512, 256)
        self.fc2_m = nn.Linear(256, 128)
        self.fc3_m = nn.Linear(128, zdim)
        self.fc_bn1_m = nn.BatchNorm1d(256)
        self.fc_bn2_m = nn.BatchNorm1d(128)

        # Mapping to [c], cmean
        self.fc1_v = nn.Linear(512, 256)
        self.fc2_v = nn.Linear(256, 128)
        self.fc3_v = nn.Linear(128, zdim)
        self.fc_bn1_v = nn.BatchNorm1d(256)
        self.fc_bn2_v = nn.BatchNorm1d(128)

    def forward(self, x):
        # x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)

        m = F.relu(self.fc_bn1_m(self.fc1_m(x)))
        m = F.relu(self.fc_bn2_m(self.fc2_m(m)))
        m = self.fc3_m(m)
        v = F.relu(self.fc_bn1_v(self.fc1_v(x)))
        v = F.relu(self.fc_bn2_v(self.fc2_v(v)))
        v = self.fc3_v(v)
        v = F.softplus(v) + 1e-8

        return m, v

class Decoder(nn.Module):
    def __init__(self,zdim,n_point,point_dim):
        super(Decoder,self).__init__()
        self.zdim = zdim
        self.n_point = n_point
        self.point_dim = point_dim
        self.n_point_3 = self.point_dim * self.n_point
        self.fc1 = nn.Linear(self.zdim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.final = nn.Linear(256,self.n_point_3)
    
    def forward(self,z):
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        output  =  self.final(x)
        output = output.reshape(-1,self.n_point,self.point_dim)
        return output

class BaseVAE(nn.Module):
    def __init__(self):
        super(BaseVAE, self).__init__()
        self.n_point = 6144
        self.point_dim = 3
        self.n_point_3 = self.point_dim * self.n_point 
        self.n_groups = 8
        self.g_points = int(self.n_point /self.n_groups)
        self.z_dim = 128
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = Encoder(self.z_dim,self.point_dim)
        
        self.decoder = Decoder(self.z_dim,self.n_point,self.point_dim)

        #set prior parameters of the vae model p(z)
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)
        self.chamfer_loss = ChamferDistanceL1()
    
    def forward(self, x):
        m, v = self.encoder(x)
        z =  sample_gaussian(m,v)
        decoder_input = z
        torch.cat((z,m),dim=-1) #BUGBUG: Ideally the encodings before passing to mu and sigma should be here.
        y = self.decoder(decoder_input)
        #compute KL divergence loss :
        p_m = self.z_prior[0].expand(m.size())
        p_v = self.z_prior[1].expand(v.size())
        kl_loss = kl_normal(m,v,p_m,p_v)
        #compute reconstruction loss 
        x_reconst = self.chamfer_loss(y,x)
        # mean or sum
        x_reconst = x_reconst.mean()
        kl_loss = kl_loss.mean()
        nelbo = x_reconst + kl_loss
        
        return nelbo, kl_loss, x_reconst, y
    
    def sample_point(self,batch):
        p_m = self.z_prior[0].expand(batch,self.z_dim).to(self.device)
        p_v = self.z_prior[1].expand(batch,self.z_dim).to(self.device)
        z =  sample_gaussian(p_m,p_v)
        decoder_input = z
        torch.cat((z,p_m),dim=-1) #BUGBUG: Ideally the encodings before passing to mu and sigma should be here.
        y = self.decoder(decoder_input)
        return y

    def reconstruct_input(self,x):
        m, v = self.encoder(x)
        z =  sample_gaussian(m,v)
        decoder_input = z
        torch.cat((z,m),dim=-1) #BUGBUG: Ideally the encodings before passing to mu and sigma should be here.
        y = self.decoder(decoder_input)
        return y
