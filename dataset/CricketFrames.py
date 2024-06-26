import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import cv2
import glob
import os

from dataset.utils import get_numeric_sort_key

class FramesLoader(Dataset):
    def __init__(self, video_root):
        self.transforms = transforms.Compose([transforms.ToTensor(),transforms.Resize((128,64)),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        shots = os.listdir(video_root)
        self.videos,self.class_labels = [],[]
        for shot in shots:
            videos = glob.glob(os.path.join(video_root,shot,'*/'))
            for video in videos:
                frames = glob.glob(os.path.join(video,'*'))
                frames = sorted(frames, key=get_numeric_sort_key)
                self.videos.append(frames)
                self.class_labels.append(shot)
        self.corp_person = True
        self.seq_len = 20 #no of frames per video
        self.channels = 3 #channels
        self.image_size = (64,64) # person box (h,w)
        self.map_labels = {'cut_shot': 0, 'cover_drive': 1}
        self.map_labels_quality = {'bad': 0, 'good': 1}
        self.size = len(self.videos)
    def __len__(self):
        return self.size
    def __getitem__(self, index):
        video_tensor=self.load_frames(self.videos[index]) # dimension Time X H X W
        class_label_tensor = torch.tensor(self.map_labels[self.class_labels[index]])
        video_info=len(video_tensor)
        # quality_tensor = torch.tensor(self.map_labels_quality[self.qualities[index]])
        return video_tensor, class_label_tensor,video_info#, quality_tensor
    def load_frames(self,video_frames):
        frames_tensor=[self.transforms(cv2.cvtColor(cv2.imread(video_frame,cv2.IMREAD_UNCHANGED),cv2.COLOR_BGR2RGB)) for video_frame in video_frames]
        return torch.stack(frames_tensor)
