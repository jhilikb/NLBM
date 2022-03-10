from __future__ import print_function
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F






from torchvision import models

     

class Vgg16c(torch.nn.Module):
    def __init__(self):
        super(Vgg16c, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        modified_pretrained = nn.Sequential(*list(vgg_pretrained_features.children())[:-1])
        for param in modified_pretrained.parameters():
            param.requires_grad = False
        self.features = modified_pretrained


        self.up = nn.PixelShuffle(2)

        self.filter1a = NL_Conv3(ksize=3, in_ch=8, out_ch=8)
        self.filter1b = NL_Conv3(ksize=3, in_ch=16, out_ch=4)
        self.filter1c = NL_Conv3(ksize=3, in_ch=20, out_ch=16)

        self.filter3a = NL_Conv3N(ksize=3, in_ch=8, out_ch=8)
        self.filter3b = NL_Conv3N(ksize=3, in_ch=16, out_ch=4)
        self.filter3c = NL_Conv3N(ksize=3, in_ch=20, out_ch=16)

        self.sk1 = nn.Conv2d(512, 8, 1)
        self.sk2 = nn.Conv2d(8, 512, 1)

        self.classifier2 = nn.Conv2d(512, 8, 1)
        self.skclassifier2 = nn.Conv2d(8, 256, 1)

        self.classifier3 = nn.Conv2d(256, 8, 1)
        self.skclassifier3 = nn.Conv2d(8, 128, 1)

        self.classifier4 = nn.Conv2d(128, 8, 1)
        self.skclassifier4 = nn.Conv2d(8, 64, 1)
       
        self.classifier5 = nn.Conv2d(64, 1, 1, 1)
        self.c1 = nn.Sequential(*list(vgg_pretrained_features.children())[:-8])
        self.c2 = nn.Sequential(*list(vgg_pretrained_features.children())[:-15])
        self.c3 = nn.Sequential(*list(vgg_pretrained_features.children())[:-22])
        self.c4 = nn.Sequential(*list(vgg_pretrained_features.children())[:-27])

    def nlcn(self,x):
        x1 = self.filter1a(x)  # 8
        x1t = torch.cat((x, x1), dim=1)  # 16
        x1 = self.filter1b(x1t)  # 4
        x1t = torch.cat((x1t, x1), dim=1)  # 20
        x1 = self.filter1c(x1t)  # 16
        # x1t = torch.cat((x1t,x1),dim=1) # 16
        # x1 = self.up(x1t) #4

        x2 = self.filter3a(x)  # 8
        x2t = torch.cat((x, x2), dim=1)  # 16
        x2 = self.filter3b(x2t)  # 4
        x2t = torch.cat((x2t, x2), dim=1)  # 20
        x2 = self.filter3c(x2t)  # 16
        # x2t = torch.cat((x2t,x2),dim=1) # 16
        # x2 = self.up(x2t) #4

        x = torch.cat((x1, x2), dim=1)  # 32
        x = self.up(x)  # 8
        return x

    def forward(self, x):
        xc1 = self.c1(x)
        xc2 = self.c2(x)
        xc3 = self.c3(x)
        xc4 = self.c4(x)
        # print('xc1:',xc1.shape)
        # print('xc2:',xc2.shape)
        # print('xc3:',xc3.shape)
        # print('xc4:',xc4.shape)
        # print("........")
        # print('Input:',x.shape)

        x = self.features(x)
        # print('Features:',x.shape)
        x = self.sk1(x)
        # print('after sk1:',x.shape)

        x = self.nlcn(x)
        # print('after nlcn:',x.shape)

        x = self.sk2(x)
        # print('after sk2:',x.shape)

        # x = self.classifier1(x)
        x = self.classifier2(x + xc1)
        x = self.nlcn(x)
        # print('after classifier2(xc1 added) and nlcn:',x.shape)
        x = self.skclassifier2(x)
        # print('after skclassifier2:',x.shape)

        x = self.classifier3(x + xc2)
        x = self.nlcn(x)
        # print('after classifier3(xc2 added) and nlcn:',x.shape)
        x = self.skclassifier3(x)
        # print('after skclassifier3:',x.shape)

        x = self.classifier4(x + xc3)
        
        x = self.nlcn(x)
        # print('after classifier4(xc3 added) and nlcn:',x.shape)
        x = self.skclassifier4(x)
        # print('after skclassifier4:',x.shape)

        x = self.classifier5(x + xc4)
        # print('after classifier5(xc4 added) :',x.shape)



        # return x1+x2
        return x
class NL_Conv3(nn.Module):
    """NON LInear Convolution Layer"""
    
    def __init__(self,ksize,in_ch,out_ch):
        super(NL_Conv3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch,ksize*ksize*in_ch*out_ch , kernel_size=ksize, padding=ksize//2, bias=False),
            nn.ReLU()
        )#ksize*ksize*out_ch*in_ch
        self.ksize= ksize
        self.in_ch= in_ch
        self.out_ch= out_ch
        self.por= ksize*ksize*in_ch
        
    def forward(self, x):
        dims=x.shape
        xc=torch.clone(x)  # Initialize xc as several copy of x
        for i in range(self.ksize*self.ksize-1):
            xc=torch.cat((xc,x),dim=1)
        
        ind=0
        for i in range(-(self.ksize//2),self.ksize//2+1):
            for j in range(-(self.ksize//2),self.ksize//2+1):
#                 tmp=x.roll(i,-1).roll(j,-2).view(dims[0],1,dims[2],dims[3])
#                 xc[:,ind,:,:]=tmp[:,0,:,:]
                  xc[:,ind*self.in_ch:(ind+1)*self.in_ch,:,:]=\
                      x.roll(i,-1).roll(j,-2).view(dims[0],self.in_ch,dims[2],dims[3])\
                      [:,0:self.in_ch,:,:]
                  ind=ind+1
        w=self.conv(x)+.0001
        
#         out=torch.clone(xc).narrow(1,0,self.out_ch)
        out=torch.empty(dims[0],self.out_ch,dims[2],dims[3]).to(xc.device)
        for i in range(self.out_ch):
            w_por=w[:,i*self.por:(i+1)*self.por,:,:]
            w_sum=torch.sum(w_por,dim=1).view(-1,1,dims[2],dims[3])
            w_norm=w_por/w_sum # normalization along Dim=1
            xp=w_norm*xc
            x1=torch.sum(xp,dim=1).view(-1,1,dims[2],dims[3])

            out[:,i:i+1,:,:]=x1.view(-1,1,dims[2],dims[3])
            
        return out

class NL_Conv3N(nn.Module):
    """NON LInear Convolution Layer"""
    
    def __init__(self,ksize,in_ch,out_ch):
        super(NL_Conv3N, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch,ksize*ksize*out_ch*in_ch , kernel_size=ksize, padding=ksize//2, bias=False)
#             nn.Hardtanh()
        )#ksize*ksize*out_ch*in_ch
        self.ksize = ksize
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.por = ksize*ksize*in_ch
        
    def forward(self, x):
        dims=x.shape
        xc=torch.clone(x)  # Initialize xc as several copy of x
        for i in range(self.ksize*self.ksize-1):
            xc=torch.cat((xc,x),dim=1)
        
        ind=0
        for i in range(-(self.ksize//2),self.ksize//2+1):
            for j in range(-(self.ksize//2),self.ksize//2+1):
                  xc[:,ind*self.in_ch:(ind+1)*self.in_ch,:,:]=\
                      x.roll(i,-1).roll(j,-2).view(dims[0],self.in_ch,dims[2],dims[3])\
                      [:,0:self.in_ch,:,:]
                  ind=ind+1
        w=self.conv(x)
        w=torch.sign(w)*(torch.abs(w)+.0001)

        out=torch.empty(dims[0],self.out_ch,dims[2],dims[3]).to(xc.device)
        for i in range(self.out_ch):
            w_por=w[:,i*self.por:(i+1)*self.por,:,:]
            w_sum=torch.sum(torch.abs(w_por),dim=1).view(-1,1,dims[2],dims[3])
            w_norm=w_por/w_sum # normalization along Dim=1
            xp=w_norm*xc
            x1=torch.sum(xp,dim=1).view(-1,1,dims[2],dims[3])

            out[:,i:i+1,:,:]=x1.view(-1,1,dims[2],dims[3])
            
        return out

