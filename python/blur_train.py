from __future__ import print_function
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import sys
import matplotlib.pyplot as plt
import scipy.io as sio
import cv2
from PIL import Image,ImageOps
import numpy as np
import argparse
from timeit import default_timer as timer
from model.vgg_nlbm_cuhk import Vgg16c
import os
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.backends import cudnn
from torch.autograd import Variable
import torchvision.utils as vutils
#from tensorboardX import SummaryWriter
from tqdm import tqdm
from math import log
# import contextual_loss as cl



from collections import namedtuple

from torchvision import models
from random import randrange



to_tensor = transforms.Compose([
    transforms.ToTensor()
])
ssize=(384,384)#384 242

class DLibdata:

    
    def __init__(self):
        
        data_dir = datadir+'/imageofblur'

        
        self.camera = data_dir
        
        self.phone_dir = datadir+'/bluroutmaps'

        # self.image_names = os.listdir(self.phone_dir)
        self.image_names = os.listdir(data_dir)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        img_name = self.image_names[idx]
        img_name2 = img_name[:-4]+'_blur.mat'
        mat = sio.loadmat(os.path.join(self.phone_dir, img_name2))
        # print(img_name2)
        # print(mat)
        phone_image = mat['final_map']#.convert('L')
        camera_image = Image.open(os.path.join(self.camera, img_name))#.convert('L')
        
        newsize=ssize
        
        # print(phone_image.shape)
        camera_image = camera_image.resize(newsize)
        
        phone_image = cv2.resize(phone_image, newsize, interpolation = cv2.INTER_NEAREST) 
        phone_image = torch.from_numpy(phone_image)
        camera_image = to_tensor(camera_image)
        # camera_image  = camera_image .unsqueeze(0)
        # phone_image  = phone_image .unsqueeze(0)
        # print(phone_image.shape,camera_image.shape)

        return  camera_image,phone_image.float()



  
def checkpoint(state, epoch):
    """Save checkpoint"""
    model_out_path = checkpoint_dir+'/model.pth'
    torch.save(state, model_out_path)
    print('Checkpoint saved to {}'.format(model_out_path))

t10=torch.Tensor([10])
t10=t10.cuda()
def train(epoch):
    """
    Train CapsuleNet model on training set

    Args:
        model: The CapsuleNet model.
        data_loader: An interator over the dataset. It combines a dataset and a sampler.
        optimizer: Optimization algorithm.
        epoch: Current epoch.
    """
    print('===> Training mode')

    num_batches = len(train_loader)# iteration per epoch. e.g: 469
    total_step = epochs * num_batches
    epoch_tot_acc = 0
    
    # Switch to train mode
    model.train()


    # start_time = timer()

    for batch_idx, (data, target) in enumerate(tqdm(train_loader, unit='batch')):
        batch_size = data.size(0)
        global_step = batch_idx + (epoch * num_batches) - num_batches

        
        

        data, target = Variable(data,requires_grad=True), Variable(target,requires_grad=True)

        if cuda_enabled:

            data = data.cuda()
            target = target.unsqueeze(0).cuda()


        # Train step - forward, backward and optimize
        optimizer.zero_grad()
        output = model(data) # output from DigitCaps (out_digit_caps)
        # print(output.shape,target.shape)
        # output3d = torch.cat((output, output), 1)
        # output3d = torch.cat((output3d, output), 1)
        # tar3d = torch.cat((target, target), 0)
        # tar3d = torch.cat((tar3d, target), 0)
        # tar3d=tar3d.unsqueeze(0)
        # print(output3d.shape,tar3d.shape)
        loss=10*criterion1(output, target)#+ 5* criterion1(output, target) + 2*criterionc(output3d*255, tar3d*255)
        # loss=10* torch.log(1/mse)/torch.log(t10)
        # loss=loss.cuda()
        # loss=Variable(loss,requires_grad=True)
        # print(loss)
        loss.backward()
        
        optimizer.step()
        

        # Calculate accuracy for each step and average accuracy for each epoch
        

        # TensorBoard logging
        # 1) Log the scalar values
       # writer.add_scalar('train/total_loss', loss.data[0], global_step)
        #writer.add_scalar('train/margin_loss', margin_loss.data[0], global_step)
        
        #writer.add_scalar('train/batch_accuracy', acc, global_step)
        #writer.add_scalar('train/accuracy', epoch_avg_acc, global_step)

        # 2) Log values and gradients of the parameters (histogram)
       

        # Print losses
        if batch_idx % 10 == 0:
            template = 'Epoch {}/{}, ' \
                    'Step {}/{}: ' \
                    '[Total loss: {:.6f}]' 
            tqdm.write(template.format(
                epoch,
                epochs,
                global_step,
                total_step,
                loss.item(),
                ))

    # Print time elapsed for an epoch
    
    
    
    end_time = timer()



stt = timer()
cuda1=True
epochs=100
datadir=sys.argv[1]
checkpoint_dir=sys.argv[2]

pixel_means= np.array([[[104.008, 116.669, 122.675]]])
criteriona=nn.SmoothL1Loss()
criterionb=nn.KLDivLoss()
criterion1 = nn.MSELoss()

    # Check GPU or CUDA is available
cuda_enabled = cuda1 and torch.cuda.is_available()
print(cuda_enabled)

kwargs = {'num_workers': 4,'pin_memory': True} if cuda_enabled else {}

print('loading train')
training_set = DLibdata()
train_loader = DataLoader(training_set, batch_size=1, shuffle=True, **kwargs)

print('===> Building model')

model=Vgg16c()
if cuda_enabled:
    #print('Utilize GPUs for computation')
    print('Number of GPU available', torch.cuda.device_count())
    model=model.cuda()
    cudnn.benchmark = True
    model = torch.nn.DataParallel(model)


    # Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.002)


if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

    # Set the logger
#writer = SummaryWriter()



for epoch in range(0, epochs + 1):
    
    train(epoch)
  



  #       Save model checkpoint
    checkpoint({'epoch': epoch + 1,'state_dict': model.state_dict(),'optimizer': optimizer.state_dict()}, epoch)

#writer.close()
