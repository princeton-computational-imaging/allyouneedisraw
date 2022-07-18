import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

class Trans(torch.nn.Module):
    def __init__(self):
        super(Trans, self).__init__()
       
        self.conv1 = torch.nn.Conv2d(3, 64, 3, padding='same')  
        self.conv2 = torch.nn.Conv2d(64, 64, 3, padding='same')
        self.conv3 = torch.nn.Conv2d(64, 64, 3, padding='same')
        self.conv4 = torch.nn.Conv2d(64, 64, 3, padding='same')
        self.conv5 = torch.nn.Conv2d(64, 4, 3, padding='same')

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
              
        return x