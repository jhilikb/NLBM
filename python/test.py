from __future__ import print_function
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import sys

import cv2
from PIL import Image,ImageOps
import numpy as np
import argparse
from timeit import default_timer as timer
import os
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.backends import cudnn
from torch.autograd import Variable
import torchvision.utils as vutils
#from tensorboardX import SummaryWriter

from vgg_nlbm_cuhk import Vgg16c




from collections import namedtuple

from torchvision import models
from random import randrange



to_tensor = transforms.Compose([
    transforms.ToTensor()
])
ssize=(384,384)#384 242



t10=torch.Tensor([10])
t10=t10.cuda()
datadir=sys.argv[1]
data_dir1=sys.argv[2]
modeldir=sys.arv[3]
def tests(epoch):
    """
    Evaluate model on validation set

    Args:
        model: The CapsuleNet model.
        data_loader: An interator over the dataset. It combines a dataset and a sampler.
    """
    print('===> Evaluate mode')

    # Switch to evaluate mode
    model.eval()


    data_dir = datadir
   
    
    imname = os.listdir(data_dir)
    # print(imname)
    count=0
    for i in imname:

        im = Image.open(data_dir+i)#.convert('L')
        newsize1 = ssize
        im = im.resize(newsize1)

       

        ii=i[:-4]
        with torch.no_grad():
            x = to_tensor(im)
            # print(x.shape)
            x = x.unsqueeze(0)#*255

            if cuda_enabled:
                # print('yes')
                x = x.cuda()
            day_image = model(x)
            day_image = day_image.cpu()

            imgg = day_image.detach().numpy()
            imgg = np.transpose(imgg, (0, 2, 3, 1))

            # print(img.shape,imm.shape,imgg.shape,output.shape)
            score = imgg[0].copy()
            # score = output[0].copy()
            score = (score - np.amin(score)) / (np.amax(score) - np.amin(score))


        cv2.imwrite(data_dir1 +'/'+ i,np.uint8(score*255))
        count=count+1


    


        
        
    


cuda1=True


    # Check GPU or CUDA is available
cuda_enabled = cuda1 and torch.cuda.is_available()
print(cuda_enabled)

kwargs = {'num_workers': 4,'pin_memory': True} if cuda_enabled else {}


model=Vgg16c()
if cuda_enabled:
    #print('Utilize GPUs for computation')
    print('Number of GPU available', torch.cuda.device_count())
    model=model.cuda()
    cudnn.benchmark = True
    model = torch.nn.DataParallel(model)

mcheckpoint = torch.load(modeldir+'/model.pth')
model.load_state_dict(mcheckpoint['state_dict'])




tests( 0)

