import torch
from torch import nn

class FractureModel(nn.Module):
    def __init__(self,input_shape:int, output_shape:int, hidden_units:int=8):
        super().__init__()
        
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units),
            nn.MaxPool2d(2,2),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units*2),
            nn.MaxPool2d(2,2),
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(in_channels=hidden_units*2, out_channels=hidden_units*3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units*3),
            nn.MaxPool2d(2,2),
            nn.Conv2d(in_channels=hidden_units*3, out_channels=hidden_units*4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units*4),
            nn.MaxPool2d(2,2),
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(in_channels=hidden_units*4, out_channels=hidden_units*4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units*4),
            nn.MaxPool2d(2,2),
        )
        self.classifier=nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*4*7*7,out_features=120),
            nn.ReLU(),
            nn.BatchNorm1d(120),
            nn.Dropout(0.3),
            nn.Linear(in_features=120,out_features=output_shape)
        )
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
            return self.classifier(self.conv3(self.conv2(self.conv1(x))))
        
    def predict(self, x:torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))