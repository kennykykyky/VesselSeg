import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class Binary(nn.Module):
    def __init__(self, th = 0.5, gamma=20):
        super(Binary, self).__init__()

        self.register_buffer('th',torch.FloatTensor([th]))
        self.register_buffer('gamma',torch.FloatTensor([gamma]))
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        
        y = x - self.th
        y = y * self.gamma
        y = self.sigmoid(y)
        
        return y 

class Sobel_3D(nn.Module):
    def __init__(self, smooth_type = 'Sobel'):
        super(Sobel_3D, self).__init__()
        
        self.smooth_type = smooth_type
        self.register_buffer('eps',torch.FloatTensor([1.0e-6]))

        # The first order difference kernel
        h1_p = np.array([[[1,0,-1]]])
        
        # The smoothing kernel
        if self.smooth_type == 'Sobel':
            h1 = np.array([[[1,2,1]]])
        elif self.smooth_type == 'Scharr_1':
            h1 = np.array([[[3,10,3]]])
        elif self.smooth_type == 'Scharr_2':
            h1 = np.array([[[47,162,47]]])

        h1 = (1/np.sum(h1))*h1         
        h2 = h1.transpose(1,2,0)
        h2_p = h1_p.transpose(1,2,0)
        h3 = h1.transpose(2,0,1)
        h3_p = h1_p.transpose(2,0,1)
        G1 = np.kron(np.kron(h2,h3),h1_p)
        G2 = np.kron(np.kron(h3,h1),h2_p)
        G3 = np.kron(np.kron(h1,h2),h3_p)
        G_1 = torch.FloatTensor(G1)
        G_2 = torch.FloatTensor(G2)
        G_3 = torch.FloatTensor(G3)

        G = torch.cat([G_1.unsqueeze(0),G_2.unsqueeze(0),G_3.unsqueeze(0)],0)
        G = G.unsqueeze(1)
        
        self.filter_3d = nn.Conv3d(in_channels=1,out_channels=3,kernel_size=3,padding=1,bias=False)
        self.filter_3d.weight = nn.Parameter(G, requires_grad=False)
        #self.binary = Binary(th=0.5,gamma = 200)


    def forward(self,img_3d):
        
        y = self.filter_3d(img_3d)
        y = torch.mul(y, y)
        y = torch.sum(y, dim=1, keepdim=True)
        y = torch.sqrt(y + self.eps)
        return y
        
class Laplacian_3D(nn.Module):
    def __init__(self, laplacian_type = '27-point'):
        super(Laplacian_3D, self).__init__()
        
        self.laplacian_type = laplacian_type
        
        if self.laplacian_type == '7-point':
            G1 = np.array([[[0,0,0],[0,1,0],[0,0,0]],
                          [[0,1,0],[1,-6,1],[0,1,0]],
                          [[0,0,0],[0,1,0],[0,0,0]]])
        elif self.laplacian_type == '19-point':
            G1 = (1/6)*np.array([[[0,1,0],[1,2,1],[0,1,0]],
                          [[1,2,1],[2,-24,2],[1,2,1]],
                          [[0,1,0],[1,2,1],[0,1,0]]])
        elif self.laplacian_type == '27-point':
            G1 = (1/26)*np.array([[[2,3,2],[3,6,3],[2,3,2]],
                          [[3,6,3],[6,-88,6],[3,6,3]],
                          [[2,3,2],[3,6,3],[2,3,2]]])

        G = torch.FloatTensor(G1)
        G = G.unsqueeze(0)
        G = G.unsqueeze(1)

        self.filter_3d = nn.Conv3d(in_channels=1,out_channels=1,kernel_size=3,padding=1,bias=False)
        self.filter_3d.weight = nn.Parameter(G, requires_grad=False)
        self.binary = Binary(th=0.5,gamma = 200)

    def forward(self,img_3d):
        
        y = self.filter_3d(img_3d)
        y = self.binary(y)
        return y

class CDT_3D(nn.Module):
    def __init__(self, k=5, gamma = 0.1):
        super(CDT_3D, self).__init__()
        '''
        3D Convolutional Distance Transform
        '''
        self.k = k
        self.register_buffer('gamma',torch.FloatTensor([gamma]))
        #self.register_buffer('eps',torch.FloatTensor([np.exp(-np.sqrt(3*k**2/4)/gamma)]))
        self.register_buffer('eps',torch.FloatTensor([1.0e-6]))
        
        dist = np.sqrt(np.sum((np.mgrid[0:self.k,0:self.k,0:self.k]-np.floor(self.k/2))**2,axis=0))
        krnl = torch.FloatTensor(np.exp(-dist/gamma))
        krnl = krnl.unsqueeze(0).unsqueeze(0)

        self.filter_3d = nn.Conv3d(in_channels=1,out_channels=1,kernel_size=self.k,padding=int(np.floor(self.k/2)),bias=False)
        self.filter_3d.weight = nn.Parameter(krnl, requires_grad=False)

    def forward(self,img_3d):
        
        y = self.filter_3d(img_3d)
        y = torch.log(y + self.eps)
        y = - self.gamma * y
        
        return y

class DT_Loss(nn.Module):
    # Include sigmoid function!
    def __init__(self, sample_weight=None, k_dt=7):
        super(DT_Loss, self).__init__()
        
        self.k_dt = k_dt
        self.register_buffer('one',torch.FloatTensor([1.00]))
        self.register_buffer('eps',torch.FloatTensor([1.0e-6]))
        self.sample_weight = sample_weight
        
        self.sigmoid = nn.Sigmoid()
        self.binary = Binary(th=0.5,gamma = 200)
    
        self.dt1 = CDT_3D(k=self.k_dt)
        self.dt2 = CDT_3D(k=self.k_dt)
        self.dt3 = CDT_3D(k=self.k_dt)
        self.dt4 = CDT_3D(k=self.k_dt)

    def forward(self,y_out,y_tg):

        y_p = self.sigmoid(y_out)
        y_p_inv = self.one - y_p
        y = self.binary(y_p)
        y_inv = self.binary(y_p_inv)
        y_tg_inv = self.one - y_tg
        z = self.dt1(y)
        z_tg = self.dt2(y_tg)
        z_inv = self.dt3(y_inv)
        z_tg_inv = self.dt4(y_tg_inv)  

        l1 = torch.mean(y_tg * z)/torch.mean(y_tg+self.eps)
        l2 = torch.mean(y_p * z_tg)/torch.mean(y_p+self.eps)
        l3 = torch.mean(y_tg_inv * z_inv)/torch.mean(y_tg_inv+self.eps)
        l4 = torch.mean(y_p_inv * z_tg_inv)/torch.mean(y_p_inv+self.eps)
        loss_pt = l1 + l2 + l3 + l4

        if not (self.sample_weight==None):
            loss = torch.mean(self.sample_weight*loss_pt)/torch.mean(self.sample_weight)
        else:
            loss = torch.mean(loss_pt)
        
        return loss

class CE_Loss(nn.Module):
    # Include sigmoid function!
    def __init__(self, sample_weight=None, pos_weight=1.00):
        super(CE_Loss, self).__init__()
        
        self.register_buffer('pos_weight',torch.FloatTensor([pos_weight]))
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none', pos_weight=self.pos_weight)
        self.sample_weight = sample_weight

    def forward(self,y_out,y_tg):
        
        loss_pt = self.bce_loss(y_out,y_tg)
        loss_pt_mean = torch.mean(loss_pt, dim=(1,2,3))
        if not (self.sample_weight==None):
            loss = torch.mean(self.sample_weight*loss_pt_mean)/torch.mean(self.sample_weight)
        else:
            loss = torch.mean(loss_pt_mean)
        
        return loss

class Dice_Loss(nn.Module):
    # Include sigmoid function!
    def __init__(self,  sample_weight=None):
        super(Dice_Loss, self).__init__()
        
        self.register_buffer('eps',torch.FloatTensor([1.0e-6]))
        self.register_buffer('one',torch.FloatTensor([1.00]))
        self.register_buffer('two',torch.FloatTensor([2.00]))
        self.sigmoid = nn.Sigmoid()
        self.sample_weight = sample_weight
        
    def forward(self,y_out,y_tg):
        
        y = self.sigmoid(y_out)
        intersection = torch.sum(y * y_tg, dim=(1,2,3))
        cardinality = torch.sum(y + y_tg, dim=(1,2,3))
        d_loss = self.one - self.two * intersection /(cardinality + self.eps)
        if not (self.sample_weight==None):
            loss = torch.mean(self.sample_weight*d_loss)/torch.mean(self.sample_weight)
        else:
            loss = torch.mean(d_loss)
        
        return loss

class ACE_Loss(nn.Module):
    # Include sigmoid function!
    def __init__(self, sample_weight=None, k_dt = 7, gamma = 0.1, b_dist = 3):
        super(ACE_Loss, self).__init__()

        self.k_dt = k_dt
        self.register_buffer('eps',torch.FloatTensor([1.0e-6]))
        self.register_parameter('gamma',nn.Parameter(torch.FloatTensor([gamma])))
        self.register_buffer('b_dist',torch.FloatTensor([b_dist]))
        self.sample_weight = sample_weight

        self.sigmoid = nn.Sigmoid()
        self.binary1 = Binary(th=0.5,gamma = 200)
        self.edge_detector = Laplacian_3D(laplacian_type='27-point')
        self.sobel = Sobel_3D(smooth_type='Scharr_2')
        self.binary2 = Binary(th=0,gamma = 200)
        self.binary3 = Binary(th=0.1,gamma = 200)
        self.relu = nn.ReLU()
        self.dt = CDT_3D(k=self.k_dt)       

    def forward(self,x_in,y_out):
        # x_in is the input img3d and y_out is the output of the network
        y = self.sigmoid(y_out)
        y = self.binary1(y)
        y1 = self.edge_detector(y)
        y2 = self.sobel(y)
        y3 = self.dt(y)
        y3 = self.b_dist - y3
        y3 = self.relu(y3)
        y4 = (self.b_dist-y3) * self.binary2(y3)
        y4 = self.binary3(y4)
        l1 = torch.mean(x_in*y,dim=(1,2,3,4))/torch.mean(y+self.eps,dim=(1,2,3,4))
        l2 = torch.mean(x_in*y4,dim=(1,2,3,4))/torch.mean(y4+self.eps,dim=(1,2,3,4))

        E1 = torch.mean(y*(x_in-l1[:,None,None,None,None])**2)/torch.mean(y+self.eps)
        E2 = torch.mean(y4*(x_in-l2[:,None,None,None,None])**2)/torch.mean(y4+self.eps)
        S = torch.mean(y1*y2)/torch.mean(y1+self.eps)
        loss_pt_mean = E1 + E2 + self.gamma*S

        if not (self.sample_weight==None):
            loss = torch.mean(self.sample_weight*loss_pt_mean)/torch.mean(self.sample_weight)
        else:
            loss = torch.mean(loss_pt_mean)
        return loss

class LumenLoss(nn.Module):
    def __init__(self, sample_weight=None, pos_weight =1.00, w_ce = 1, w_dt = 0.1, w_ace = 0.1, w_dice = 1):
        super(LumenLoss, self).__init__()

        self.sample_weight = sample_weight
        self.pos_weight = pos_weight
        self.register_buffer('w_ce',torch.FloatTensor([w_ce]))
        self.register_buffer('w_dt',torch.FloatTensor([w_dt]))
        self.register_buffer('w_ace',torch.FloatTensor([w_ace]))
        self.register_buffer('w_dice',torch.FloatTensor([w_dice]))

        self.ce_loss = CE_Loss(sample_weight = self.sample_weight,pos_weight = self.pos_weight)
        self.dice_loss = Dice_Loss(sample_weight = self.sample_weight)
        self.dt_loss = DT_Loss(sample_weight = self.sample_weight, k_dt=7)
        self.ace_loss = ACE_Loss(sample_weight = self.sample_weight, k_dt=7, gamma = 0.1, b_dist = 3)

    def forward(self,x_in, y_out,y_tg, w_smp):

        self.ce_loss.sample_weight = w_smp
        self.dice_loss.sample_weight = w_smp
        self.dt_loss.sample_weight = w_smp
        self.ce_loss.sample_weight = w_smp
        loss1 = self.ce_loss(y_out,y_tg)
        loss2 = self.dice_loss(y_out,y_tg)
        loss3 = self.dt_loss(y_out,y_tg)
        loss4 = self.ace_loss(x_in,y_out)
        loss = self.w_ce * loss1 + loss2 + self.w_dt * loss3 + self.w_ace * loss4

        # make the shape of the tensor from [1] to [0]
        loss = torch.mean(loss)
        
        return loss

       