## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool1 = nn.MaxPool2d(2,2)
        self.drop1 = nn.Dropout(0.2)
        # output size = (W - F)/S +1 = (225 - 5)/1 +1 = 220        
        # 220/2 = 110 the output tensor would be: (32, 110, 110)
        
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool2 = nn.MaxPool2d(2,2)
        self.drop2 = nn.Dropout(0.2)
        # output size = (W - F)/S +1 = (110 - 3)/1 +1 = 108
        # 108/2 = 54 the output tensor would be: (64, 54, 54)
        
        
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.pool3 = nn.MaxPool2d(2,2)
        self.drop3 = nn.Dropout(0.25)
        # output size = (W - F)/S +1 = (54 - 3)/1 +1 = 52        
        # 52/2 = 26 the output tensor would be: (128, 26, 26)
        
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.pool4 = nn.MaxPool2d(2,2)
        self.drop4 = nn.Dropout(0.25)
        # output size = (W - F)/S +1 = (26 - 3)/1 +1 = 24
        # 24/2 = 12 the output tensor would be: (256, 12, 12)
        
        
        self.conv5 = nn.Conv2d(256, 512, 3)
        self.pool5 = nn.MaxPool2d(2,2)
        self.drop5 = nn.Dropout(0.3)
        # output size = (W - F)/S +1 = (12 - 3)/1 +1 = 10 
        # 10/2 = 5 the output tensor would be: (512, 5, 5)
        
        
        self.conv6 = nn.Conv2d(512, 1024, 3)
        self.pool6 = nn.MaxPool2d(2,2)
        self.drop6 = nn.Dropout(0.3)
        # output size = (W - F)/S +1 = (5 - 1)/1 +1 = 5
        # 5/2 = 2 the output tensor would be: (1024, 2, 2)
        
               
        self.conv7 = nn.Conv2d(1024,136,1)
#         self.drop6 = nn.Dropout(0.4)
#         self.dense2 = nn.Linear( 1024,136)
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.drop1(self.pool1(F.elu(self.conv1(x))))
        x = self.drop2(self.pool2(F.elu(self.conv2(x))))
        x = self.drop3(self.pool3(F.elu(self.conv3(x))))
        x = self.drop4(self.pool4(F.elu(self.conv4(x))))        
        x = self.drop5(self.pool5(F.elu(self.conv5(x))))
        x = self.drop6(self.pool6(F.elu(self.conv6(x))))
        
        x = F.elu(self.conv7(x))
        x = x.view(-1, 136)
#         x = self.drop6(x)
#         x = F.tanh(self.dense2(x))
        
        return x
