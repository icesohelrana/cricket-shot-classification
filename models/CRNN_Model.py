from torch import nn
from torchvision import models
import torch

class Resnt18Rnn(nn.Module):
    '''
    CRNN Model for action recognition/ shot classification.
    It has two stages
        1. CNN feature extractor on frames
        2. Feeding CNN features to RNN/LSTM/GRU model
    '''
    def __init__(self, num_classes, rnn_hidden_size, rnn_num_layers, dr_rate=0.1,pretrained=True):
        '''
        num_classes: Number of classes in the dataset.
        rnn_hidden_size: Hidden size of LSTM
        rnn_num_layers: Stack of RNN/LSTM Layers
        dr_rate: Dropout rate
        pretrained: Pretrained weights for CNN backbone (Default = True)
        '''
        super(Resnt18Rnn, self).__init__()
        # num_classes
        # dr_rate
        # pretrained
        # rnn_hidden_size
        # rnn_num_layers
        baseModel = models.resnet18(pretrained=pretrained)
        num_features = baseModel.fc.in_features #get feature size extracted from CNN
        baseModel.fc = Identity()
        self.baseModel = baseModel
        self.dropout = nn.Dropout(dr_rate)
        self.rnn = nn.LSTM(num_features, rnn_hidden_size, rnn_num_layers)
        self.fc1 = nn.Linear(rnn_hidden_size, num_classes)

    def forward(self, x):
        b_z, ts, c, h, w = x.shape
        #b_z = batch, ts=#frames, c=channels(RGB), h,w of image
        ii = 0
        y = self.baseModel((x[:, ii])) #give b_s*512 and get h0, c0
        output, (hn, cn) = self.rnn(y.unsqueeze(1))
        for ii in range(1, ts):
            y = self.baseModel((x[:, ii])) # get CNN features
            out, _ = self.rnn(y.unsqueeze(1), (hn, cn)) # Pass CNN features to RNN
        feat = self.dropout(out[:, -1]) #apply dropot
        out = self.fc1(feat) #Get Final predictions
        return out #[b_z, number of classes]

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

# if __name__ == '__main__':
#     '''
#     You can always debug the model from thee main method
#     '''
#     #define sample dataset
#     input = torch.rand((8, 20, 3, 256, 128)) #(bs, 20, 3, w, h) ##frames < 20 # pad with zeros
#     num_classes = 2
#     rnn_hidden_size = 256
#     rnn_num_layers = 2
#     model = Resnt18Rnn(num_classes=2, rnn_hidden_size=rnn_hidden_size, rnn_num_layers=2)
#     #Collect the output
#     path = "/home/vishnu/projects/ienhance/models/22.pth"
#     model.load_state_dict(torch.load(path))
#     y = model(input)
#     model.eval()
#     print(y.shape)