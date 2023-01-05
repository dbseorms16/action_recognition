import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from vivit2 import ViViT,ClassificationHead
import pytorchvideo.data
import torch
import torch.optim as optim
import torch
from torchmetrics import Accuracy
from typing import Any, Callable, Dict, Optional, Type
import torch
from pytorchvideo.data.clip_sampling import ClipSampler
import pathlib

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

num_classes = 101
dataset_root_path = '/home/hong/workspace/datasets/UCF-101/ii/'
dataset_root_path = pathlib.Path(dataset_root_path)
video_count_train = len(list(dataset_root_path.glob("train/*/*.avi")))
video_count_val = len(list(dataset_root_path.glob("val/*/*.avi")))
video_count_test = len(list(dataset_root_path.glob("test/*/*.avi")))
video_total = video_count_train + video_count_val + video_count_test
print(f"Total videos: {video_total}")

all_video_file_paths = (
    list(dataset_root_path.glob("train/*/*.avi"))
    + list(dataset_root_path.glob("val/*/*.avi"))
    + list(dataset_root_path.glob("test/*/*.avi"))
)
print(all_video_file_paths[:5])

class_labels = sorted({str(path).split("/")[-2] for path in all_video_file_paths})
print(class_labels)
num_classes = 101
label2id = {label: i for i, label in enumerate(class_labels)}
id2label = {i: label for label, i in label2id.items()}

print(f"Unique classes: {list(label2id.keys())}.")

mean = [0.485, 0.456, 0.406] 
std = [0.229, 0.224, 0.225] 
resize_to = 224
num_frames_to_sample = 16
sample_rate = 4
fps = 30

# _CLIP_DURATION = clip_duration  # Duration of sampled clip for each video
# clip_duration = 4
_BATCH_SIZE = 1
_NUM_WORKERS = 8  # Number of parallel processes fetching data
clip_duration = num_frames_to_sample * sample_rate / fps

import os
class UCF101DataModule():
    # Training dataset transformations.
    def train_dataloader(self):
        train_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(num_frames_to_sample),
                            Lambda(lambda x: x / 255.0),
                            Normalize(mean, std),
                            RandomShortSideScale(min_size=256, max_size=320),
                            RandomCrop(resize_to),
                            RandomHorizontalFlip(p=0.5),
                        ]
                    ),
                ),
            ]
        )
        # Training dataset.
        train_dataset = pytorchvideo.data.Ucf101(
            data_path=os.path.join(dataset_root_path, "train"),
            clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
            decode_audio=False,
            transform=train_transform,
        )
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=_BATCH_SIZE,
            num_workers=_NUM_WORKERS,
        )

    # Validation and evaluation datasets' transformations.
    def val_dataloader(self):
        val_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(num_frames_to_sample),
                            Lambda(lambda x: x / 255.0),
                            Normalize(mean, std),
                            Resize((resize_to, resize_to)),
                        ]
                    ),
                ),
            ]
        )

    # Validation and evaluation datasets.
        val_dataset = pytorchvideo.data.Ucf101(
            data_path=os.path.join(dataset_root_path, "val"),
            clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
            decode_audio=False,
            transform=val_transform,
        )
        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=_BATCH_SIZE,
            num_workers=_NUM_WORKERS,
        )
    def test_dataloader(self):
        val_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(num_frames_to_sample),
                            Lambda(lambda x: x / 255.0),
                            Normalize(mean, std),
                            Resize((resize_to, resize_to)),
                        ]
                    ),
                ),
            ]
        )
        test_dataset = pytorchvideo.data.Ucf101(
            data_path=os.path.join(dataset_root_path, "test"),
            clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
            decode_audio=False,
            transform=val_transform,
        )
        return torch.utils.data.DataLoader(
            test_dataset,
            batch_size=_BATCH_SIZE,
            num_workers=_NUM_WORKERS,
        )
    


class SSV2DataModule():

  # Dataset configuration
    #_DATA_PATH = dataset_root_path
    num_frames_to_sample = 16 # 인풋 프레임 수
    sample_rate = 4 # 영상 길이?
    fps = 30 #fps
    clip_duration = num_frames_to_sample * sample_rate / fps   #나눌 클립 개수 
    # _CLIP_DURATION = clip_duration  # Duration of sampled clip for each video
    # clip_duration = 4
    _BATCH_SIZE = 12
    _NUM_WORKERS = 8  # Number of parallel processes fetching data
    mean = [0.485, 0.456, 0.406] 
    std = [0.229, 0.224, 0.225] 
    resize_to = 224

    def train_dataloader(self):
        """
        Create the Kinetics train partition from the list of video labels
        in {self._DATA_PATH}/train
        """
    
        train_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(self.num_frames_to_sample),
                        Lambda(lambda x: x / 255.0),
                        Normalize(self.mean, self.std),
                        RandomShortSideScale(min_size=256, max_size=320),
                        RandomCrop(self.resize_to),
                        RandomHorizontalFlip(p=0.5),
                    ]
                ),
            ),
            ]
        )
        train_dataset = pytorchvideo.data.SSv2(
        label_name_file = "/home/hong/workspace/source/video_mae/datasets/ssv2/labels.json",
        video_label_file = "/home/hong/workspace/source/video_mae/datasets/ssv2/train.json",
        video_path_label_file = "/home/hong/workspace/source/video_mae/datasets/ssv2/train_tiny.csv",
        video_path_prefix = "/home/hong/workspace/datasets/ssv2/",
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", self.clip_duration),
        transform=train_transform,
        )
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
        )

    def val_dataloader(self):
        """
        Create the Kinetics validation partition from the list of video labels
        in {self._DATA_PATH}/val
        """
        val_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(self.num_frames_to_sample),
                        Lambda(lambda x: x / 255.0),
                        Normalize(self.mean, self.std),
                        Resize((self.resize_to, self.resize_to)),
                    ]
                ),
            ),
            ]
        )
        val_dataset = pytorchvideo.data.SSv2(
        label_name_file = "/home/hong/workspace/source/video_mae/datasets/ssv2/labels.json",
        video_label_file = "/home/hong/workspace/source/video_mae/datasets/ssv2/validation.json",
        video_path_label_file = "/home/hong/workspace/source/video_mae/datasets/ssv2/val_tiny.csv",
        video_path_prefix = "/home/hong/workspace/datasets/ssv2/",
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", self.clip_duration),
        transform=val_transform,
        )
        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
        )
    
    def test_dataloader(self):
        val_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(self.num_frames_to_sample),
                        Lambda(lambda x: x / 255.0),
                        Normalize(self.mean, self.std),
                        Resize((self.resize_to, self.resize_to)),
                    ]
                ),
            ),
            ]
        )
        test_dataset = pytorchvideo.data.SSv2(
        label_name_file = "/home/hong/workspace/source/video_mae/datasets/ssv2/labels.json",
        video_label_file = "/home/hong/workspace/source/video_mae/datasets/ssv2/validation.json",
        video_path_label_file = "/home/hong/workspace/source/video_mae/datasets/ssv2/val_tiny.csv",
        video_path_prefix = "/home/hong/workspace/datasets/ssv2/",
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", self.clip_duration),
        transform=val_transform,
        )
        return torch.utils.data.DataLoader(
        test_dataset,
        batch_size=self._BATCH_SIZE,
        num_workers=self._NUM_WORKERS,
        )


class VideoTransformerLightningModule(nn.Module):
    def __init__(self):
        super().__init__()
        #epoch=30, batch_size=4, num_workers=4, resume=False, resume_from_checkpoint=None, log_interval=30, save_ckpt_freq=20, 
        # objective='supervised', eval_metrics='finetune', gpus=-1, root_dir='./', num_class=174, num_samples_per_cls=10000, 
        # img_size=224, num_frames=16, frame_interval=16, multi_crop=False, mixup=False, auto_augment=None, arch='vivit', attention_type='fact_encoder', 
        # pretrain_pth='models/vivit_model.pth', weights_from='imagenet', seed=0, optim_type='sgd', lr_schedule='cosine', lr=7.8125e-05, layer_decay=0.75,
        # min_lr=1e-06, use_fp16=True, weight_decay=0.05, weight_decay_end=0.05, clip_grad=0, warmup_epochs=5)
        #self.model = vit_base_patch16_224()
        
        
        self.model = ViViT(
				pretrain_pth="models/vivit_model.pth",
				weights_from="imagenet",
				img_size=224,
				num_frames=16,
				attention_type="fact_encoder")
        self.cls_head = ClassificationHead(
			174, self.model.embed_dims, eval_metrics="finetune")
        
        
        
    def forward(self, x):
        preds = self.model(x)
        preds = self.cls_head(preds)
        return preds



if __name__ == "__main__":
    epoch = 1000
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #data_module = SSV2DataModule()
    data_module = UCF101DataModule()
    #train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()
    #test_dataloader = data_module.test_dataloader()
    
    train_top1_acc = Accuracy(task="multiclass",num_classes=174).to(device)
    #train_top5_acc = Accuracy(task="multiclass",top_k=5,num_classes=174).to(device)
    
    net = VideoTransformerLightningModule() 
    #net.load_state_dict(torch.load("save/best_250_0.8853.pth"))
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    best_model = None
    best_loss = 100000
    best_accuracy=-1
    val_loss = 0
    val_running_loss = 0
    accuracy = 0
    total_acc = 0  
    val_running_loss = 0.0
    top1_acc=0
    val_total=0
    val_acc=0
    print("**Run validation**")
    with torch.no_grad():
        for i, data in enumerate(val_dataloader, 0):
            inputs = data['video'].to(device)
            labels = data['label'].to(device)
    
            outputs = net(inputs)
            
            top1_acc = train_top1_acc(outputs.softmax(dim=-1), labels).item()
            
            if val_total<i:
                val_total=i
            val_acc = (top1_acc + val_acc)
        total_acc = val_acc/val_total
        total_acc = round(total_acc,4)
        print(f'[{epoch + 1}] top1: {total_acc:.3f}')
       
            
            

    print('Finished Training')








