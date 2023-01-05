import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.optim as optim
import torch
from torchmetrics import Accuracy
from typing import Any, Callable, Dict, Optional, Type
import torch
from pytorchvideo.data.clip_sampling import ClipSampler
import pytorchvideo.data

from einops import rearrange, repeat, reduce


import pathlib
import pytorchvideo
import pytorchvideo.transforms.functional
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)

# flow resize and conv
# def  flow_frame (vis_flow_preds,frame_size):          #vis_flow_preds => rad 동영상 프레임
#     count = 0
#     frame_vis_flow_preds=[]
#     frame_sum=np.zeros((len(vis_flow_preds[0]),len(vis_flow_preds[0][0])))

#     for i in range(len(vis_flow_preds)):
#         frame_sum+=vis_flow_preds[i]   #frame  rad 누적
        
#         if count >=frame_size :
#             count=0
#             frame_sum/=frame_size
#             frame_vis_flow_preds.append(frame_sum)   #
#             frame_sum=np.zeros((len(vis_flow_preds[0]),len(vis_flow_preds[0][0])))


#         count +=1

#         if i is (len(vis_flow_preds)-1):
#             frame_vis_flow_preds.append(frame_sum)    


#     #print(np.shape(frame_vis_flow_preds))
#     return frame_vis_flow_preds



def flow_frame(vis_flow_preds):
    frame_vis_flow_preds=[]

    for i in range(len(vis_flow_preds)):

        frame_vis_flow_preds.append(vis_flow_preds[i])   #flow 저장

    return frame_vis_flow_preds



def clipsampler(vis_flow_preds,patch_size=16,tube_size=2):
    frame_vis_flow_preds=flow_frame(vis_flow_preds)
    num_frames_to_sample=16
    frame_vis_flow_preds_=torch.tensor(frame_vis_flow_preds)

    frame_vis_flow_preds_= pytorchvideo.transforms.functional.uniform_temporal_subsample( frame_vis_flow_preds_, num_frames_to_sample, -3)
    frame_vis_flow_preds_= frame_vis_flow_preds_.unsqueeze(dim=0)   #channel
    frame_vis_flow_preds_= frame_vis_flow_preds_.unsqueeze(dim=0)   #batch size
    print(frame_vis_flow_preds_.shape)

    m=nn.Conv3d(1,768,kernel_size=(tube_size,patch_size,patch_size),stride=(tube_size,patch_size,patch_size))
    output = m(frame_vis_flow_preds_.float())
    output = rearrange(output, 'b c t h w -> (b t) (h w) c')
    print(output.shape)
    # clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", 2)
    # frame_vis_flow_preds_=torch.from_numpy(frame_vis_flow_preds)

    



"""


def patch_attn(vis_flow_preds,frame_size=16,patch_size=14,tube_size=2):
    rad_frame =np.array(flow_frame(vis_flow_preds,frame_size))
    ##print(rad_frame)    #전체 프레임 덩어리 3d average pooling
    rad_frame_t=torch.from_numpy(rad_frame)
    ##print((len(rad_frame_t[0])//patch_size))
    m=nn.Conv3d(1,768,kernel_size=(tube_size,patch_size,patch_size),stride=(tube_size,patch_size,patch_size))
    output=m(rad_frame_t)

    output -=output.min()
    output/=output.max()

    print(output)
    return output

def frame_attn(vis_flow_preds,frame_size):
    rad_frame =flow_frame(vis_flow_preds,frame_size)
    rad_sum=np.sum(np.sum(rad_frame,axis=1),axis=1)
    scaled_rad_sum= (rad_sum -np.min(rad_sum))/(np.max(rad_sum)-np.min(rad_sum)) #normalize
    print(scaled_rad_sum)
    return scaled_rad_sum




"""




     