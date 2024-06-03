This repo contains cricket shot classification (e.g. cut shot, cover drive) model from net session video clips. 

Model: Used a pretrained Resnet18 for feature extraction of each from of a clip and A RNN takes the features from each frame in a video. Then, the RNN head is used to predicts the class of the shot.

Dataset: We capture some videos of netsession using 30fps mobile camera and label those videos.

Create a conda environment and install all required packages, Set the parameters and data path in configs/cfg.yaml file.
Then run train.py to train and evaluate the model
