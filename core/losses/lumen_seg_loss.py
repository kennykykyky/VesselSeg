import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# sys.path.append('/v/ai/backup/shashemi/VWI_DL/Utils/')
# from wall_dataset import Wall_Dataset
from matplotlib import pyplot as plt

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

class Sobel_2D(nn.Module):
    def __init__(self, smooth_type = 'Sobel'):
        super(Sobel_2D, self).__init__()
        
        self.smooth_type = smooth_type
        self.register_buffer('eps',torch.FloatTensor([1.0e-6]))

        # The first order difference kernel
        h1_p = np.array([[1,0,-1]])
        
        # The smoothing kernel
        if self.smooth_type == 'Sobel':
            h1 = np.array([[1,2,1]])
        elif self.smooth_type == 'Scharr_1':
            h1 = np.array([[3,10,3]])
        elif self.smooth_type == 'Scharr_2':
            h1 = np.array([[47,162,47]])

        h1 = (1/np.sum(h1))*h1 
        
        h2 = h1.transpose(1,0)
        h2_p = h1_p.transpose(1,0)
        G1 = np.kron(h2,h1_p)
        G2 = np.kron(h1,h2_p)
        G_1 = torch.FloatTensor(G1)
        G_2 = torch.FloatTensor(G2)

        G = torch.cat([G_1.unsqueeze(0),G_2.unsqueeze(0)],0)
        G = G.unsqueeze(1)

        self.filter_2d = nn.Conv2d(in_channels=1,out_channels=2,kernel_size=3,padding=1,bias=False)
        self.filter_2d.weight = nn.Parameter(G, requires_grad=False)
        
    def forward(self,img_2d):
        
        y = self.filter_2d(img_2d)
        z = torch.mul(y, y)
        z = torch.sum(z, dim=1, keepdim=True)
        z = torch.sqrt(z+self.eps)
        return z

class Laplacian_2D(nn.Module):
    def __init__(self, laplacian_type = '5-point'):
        super(Laplacian_2D, self).__init__()
        
        self.laplacian_type = laplacian_type
        
        # The 3D Laplacian kernel
        if self.laplacian_type == '5-point':
            G1 = np.array([[0,1,0],[1,-4,1],[0,1,0]])
        elif self.laplacian_type == '9-point':
            G1 = (1/4)*np.array([[1,2,1],[2,-12,2],[1,2,1]])
            
        G_1 = torch.FloatTensor(G1)
        G = G_1.unsqueeze(0)
        G = G.unsqueeze(1)
        self.filter_2d = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,padding=1,bias=False)
        self.filter_2d.weight = nn.Parameter(G, requires_grad=False)
        self.binary = Binary(th=0.5,gamma = 200)
        
    def forward(self,img_2d):
        y = self.filter_2d(img_2d)
        y = self.binary(y)
        return y


class CDT_2D(nn.Module):
    def __init__(self, k=7, gamma = 0.1):
        super(CDT_2D, self).__init__()
        '''
        2D Convolutional Distance Transform
        '''
        self.k = k
        self.register_buffer('gamma',torch.FloatTensor([gamma]))
        self.register_buffer('eps',torch.FloatTensor([1.0e-6]))
        
        dist = np.sqrt(np.sum((np.mgrid[0:self.k,0:self.k]-np.floor(self.k/2))**2,axis=0))
        krnl = torch.FloatTensor(np.exp(-dist/gamma))
        krnl = krnl.unsqueeze(0).unsqueeze(0)

        self.filter_2d = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=self.k,padding=int(np.floor(self.k/2)),bias=False)
        self.filter_2d.weight = nn.Parameter(krnl, requires_grad=False)

    def forward(self,img_2d):
        
        y = self.filter_2d(img_2d)
        y = torch.log(y + self.eps)
        y = - self.gamma * y
        
        return y




class DT_Loss(nn.Module):
    def __init__(self,sample_weight=None, k_dt=7):
        super(DT_Loss, self).__init__()

        self.register_buffer('one',torch.FloatTensor([1.00]))
        self.register_buffer('eps',torch.FloatTensor([1.0e-6]))
        self.sample_weight = sample_weight
        self.k_dt = k_dt
        
        self.sigmoid = nn.Sigmoid()
        self.binary = Binary(th=0.5,gamma = 200)
        
        self.dt1 = CDT_2D(k=self.k_dt)
        self.dt2 = CDT_2D(k=self.k_dt)
        self.dt3 = CDT_2D(k=self.k_dt)
        self.dt4 = CDT_2D(k=self.k_dt)

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

        l1 = torch.mean(y_tg * z,dim=(1,2,3))/torch.mean(y_tg+self.eps,dim=(1,2,3))
        l2 = torch.mean(y_p * z_tg,dim=(1,2,3))/torch.mean(y_p+self.eps,dim=(1,2,3))
        l3 = torch.mean(y_tg_inv * z_inv,dim=(1,2,3))/torch.mean(y_tg_inv+self.eps,dim=(1,2,3))
        l4 = torch.mean(y_p_inv * z_tg_inv,dim=(1,2,3))/torch.mean(y_p_inv+self.eps,dim=(1,2,3))
        loss_pt = l1 + l2 + l3 + l4
        if not (self.sample_weight==None):
            loss = torch.mean(self.sample_weight*loss_pt)/torch.mean(self.sample_weight)
        else:
            loss = torch.mean(loss_pt)
        
        return loss




class Dice_Loss(nn.Module):
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

class CE_Loss(nn.Module):
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


class ACE_Loss(nn.Module):
    def __init__(self, sample_weight=None, k_dt = 7, gamma = 0.1, b_dist = 3):
        super(ACE_Loss, self).__init__()
        
        self.sample_weight = sample_weight
        self.k_dt = k_dt
        self.register_buffer('eps',torch.FloatTensor([1.0e-6]))
        self.register_parameter('gamma',nn.Parameter(torch.FloatTensor([gamma])))
        self.register_buffer('b_dist',torch.FloatTensor([b_dist]))

        self.sigmoid = nn.Sigmoid()
        self.binary1 = Binary(th=0.5,gamma = 200)
        self.edge_detector = Laplacian_2D(laplacian_type='9-point')
        self.sobel = Sobel_2D(smooth_type='Scharr_2')
        self.binary2 = Binary(th=0,gamma = 200)
        self.binary3 = Binary(th=0.1,gamma = 200)
        self.relu = nn.ReLU()
        self.dt = CDT_2D(k=self.k_dt)  

    def forward(self,x_in,y_out):
        y = self.sigmoid(y_out)
        y = self.binary1(y)
        y1 = self.edge_detector(y)
        y2 = self.sobel(y)
        y3 = self.dt(y)
        y3 = self.b_dist - y3
        y3 = self.relu(y3)
        y4 = (self.b_dist-y3) * self.binary2(y3)
        y4 = self.binary3(y4)
        l1 = torch.mean(x_in*y,dim=(1,2,3))/torch.mean(y+self.eps,dim=(1,2,3))
        l2 = torch.mean(x_in*y4,dim=(1,2,3))/torch.mean(y4+self.eps,dim=(1,2,3))

        E1 = torch.mean(y*(x_in-l1[:,None,None,None])**2,dim=(1,2,3))/torch.mean(y+self.eps,dim=(1,2,3))
        E2 = torch.mean(y4*(x_in-l2[:,None,None,None])**2,dim=(1,2,3))/torch.mean(y4+self.eps,dim=(1,2,3))
        S = torch.mean(y1*y2,dim=(1,2,3))/torch.mean(y1+self.eps,dim=(1,2,3))
        loss_pt_mean = E1 + E2 + self.gamma*S

        if not (self.sample_weight==None):
            loss = torch.mean(self.sample_weight*loss_pt_mean)/torch.mean(self.sample_weight)
        else:
            loss = torch.mean(loss_pt_mean)
        
        return loss





class Lumen_Seg_Loss(nn.Module):
    def __init__(self, sample_weight = None, pos_weight =1.00, w_ce = 1, w_dt = 0.1, w_ace = 0.1):
        super(Lumen_Seg_Loss, self).__init__()
        
        self.sample_weight = sample_weight
        self.pos_weight = pos_weight
        self.register_buffer('w_ce',torch.FloatTensor([w_ce]))
        self.register_buffer('w_dt',torch.FloatTensor([w_dt]))
        self.register_buffer('w_ace',torch.FloatTensor([w_ace]))


        self.ce_loss = CE_Loss(sample_weight = self.sample_weight,pos_weight = self.pos_weight)
        self.dice_loss = Dice_Loss(sample_weight = self.sample_weight)
        self.dt_loss = DT_Loss(sample_weight = self.sample_weight, k_dt=7)
        self.ace_loss = ACE_Loss(sample_weight = self.sample_weight, k_dt=7, gamma = 0.1, b_dist = 3)
        

    def forward(self,x_in, y_out,y_tg,w_smp):

        self.ce_loss.sample_weight = w_smp
        self.dice_loss.sample_weight = w_smp
        self.dt_loss.sample_weight = w_smp
        self.ce_loss.sample_weight = w_smp
        loss1 = self.ce_loss(y_out,y_tg)
        loss2 = self.dice_loss(y_out,y_tg)
        loss3 = self.dt_loss(y_out,y_tg)
        loss4 = self.ace_loss(x_in,y_out)
        loss = self.w_ce * loss1 + loss2 + self.w_dt * loss3 + self.w_ace * loss4
        return loss

def main():
    
    
    
    data_path = '/v/ai/nobackup/shashemi/VWI_DL/Wall_Segmentation_Train/debug/'
    dataset_train = Wall_Dataset(data_path, data_type = 'train', aug=True)
    train_loader = DataLoader(dataset=dataset_train, num_workers=8, batch_size=8, shuffle=True)
    
    criterion1 = Lumen_Seg_Loss(sample_weight = None, pos_weight =1.00, w_ce =1, w_dt = 0.5, w_ace=1).cuda()
    criterion2 = CE_Loss(sample_weight = None, pos_weight =1.00).cuda()
    criterion3 = Dice_Loss(sample_weight = None).cuda()
    criterion4 = DT_Loss(sample_weight = None, k_dt=7).cuda()
    criterion5 = ACE_Loss(sample_weight = None, k_dt=7, gamma = 0.1, b_dist = 3).cuda()
        

    
    
    for x, y_tg, y_tg_1, w_smp in train_loader:
        dumy_input = x[:,5:6,:,:]
        dumy_output = y_tg + torch.Tensor(torch.randn(y_tg.shape))
        dumy_output = 1.00 * (dumy_output > 0.5)
        dumy_weight = w_smp
        dumy_target = y_tg

        loss1 = criterion1(dumy_input.cuda(), dumy_output.cuda(), dumy_target.cuda(), dumy_weight.cuda())
        criterion2.sample_weight = dumy_weight.cuda()
        loss2 = criterion2(dumy_output.cuda(), dumy_target.cuda())
        criterion3.sample_weight = dumy_weight.cuda()
        loss3 = criterion3(dumy_output.cuda(), dumy_target.cuda())
        criterion4.sample_weight = dumy_weight.cuda()
        loss4 = criterion4(dumy_output.cuda(), dumy_target.cuda())
        criterion5.sample_weight = dumy_weight.cuda()
        loss5 = criterion5(dumy_input.cuda(), dumy_output.cuda())
        

        print("weight shape:")
        print(dumy_weight.shape)
        print("output shape:")
        print(dumy_output.shape)
        print('Loss:')
        print(loss1.cpu(),loss2.cpu(),loss3.cpu(),0.5*loss4.cpu(),1*loss5.cpu())
        print(loss2.cpu() + loss3.cpu() + 0.5*loss4.cpu() + 1*loss5.cpu())
    
   
if __name__ == "__main__":
    main()
       

        
        



