This repo contains cricket shot classification (e.g. cut shot, cover drive) model from net session video clips. 

Model: Used a pretrained Resnet18 for feature extraction of each from of a clip and A RNN takes the features from each frame in a video. Then, the RNN head is used to predicts the class of the shot.

## Dataset:
 We capture some videos of netsession using 30fps mobile camera and label those videos.
The folder structure of the video files:
 
* **data_dir**
    * cover_drive
        * 1.mp4
        * 2.mp4
    * cut_shot
        * 1.mp4
        * x.mp4 

Set video_dir='data_dir/cover_drive' in frame_extractor.sh
and run 

    ./frame_extractor.sh

Extract videos to frame for all type of shots.

## Train the model
Create a conda environment and install all required packages, like pytorch, torchvision, opencv etc,
 
Set the parameters and data dir in configs/cfg.yaml file.
Then run 

    python train.py

to train and display the result in tensorboard.
