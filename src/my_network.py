# PyTorch & Pytorch Lightning
from torch import nn
from torchvision.models.alexnet import AlexNet
import torch

# Blocks
from src.blocks import InceptionBlock
from src.blocks import ResidualBlock
from src.blocks import SEBlock

# [TODO: Optional] Rewrite this class if you want
class MyNetwork(AlexNet):
    def __init__(self, 
                 num_classes: int = 200,
                 dropout: float = 0.5
        ):
        super().__init__(num_classes=num_classes, dropout=dropout)

        # [TODO] Modify feature extractor part in AlexNet


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [TODO: Optional] Modify this as well if you want
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x        

class MyAlex(AlexNet):
    def __init__(self, 
                 num_classes: int = 200,
                 dropout: float = 0.5
        ):
        super().__init__(num_classes=num_classes, dropout=dropout)

        # [TODO] Modify feature extractor part in AlexNet
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),            
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),            
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

class MyAlexDeep(AlexNet):
    def __init__(self, 
                 num_classes: int = 200,
                 dropout: float = 0.5
        ):
        super().__init__(num_classes=num_classes, dropout=dropout)

        # [TODO] Modify feature extractor part in AlexNet
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    
class MyAlexBN(AlexNet):
    def __init__(self, 
                 num_classes: int = 200,
                 dropout: float = 0.5
        ):
        super().__init__(num_classes=num_classes, dropout=dropout)

        # [TODO] Modify feature extractor part in AlexNet
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            #### Add batch normalization ####
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            #### Add batch normalization ####
            nn.BatchNorm2d(192),
            
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        
class MyAlexDeepBN(AlexNet):
    def __init__(self, 
                 num_classes: int = 200,
                 dropout: float = 0.5
        ):
        super().__init__(num_classes=num_classes, dropout=dropout)

        # [TODO] Modify feature extractor part in AlexNet
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            #### Add batch normalization ####
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            #### Add batch normalization ####
            nn.BatchNorm2d(192),
            
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        
class MyAlexInception(AlexNet):
    def __init__(self, 
                 num_classes: int = 200,
                 dropout: float = 0.5
        ):
        super().__init__(num_classes=num_classes, dropout=dropout)

        # [TODO] Modify feature extractor part in AlexNet
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            #### Add inception module ####
            InceptionBlock(64),
            InceptionBlock(256),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    
class MyAlexResidual(AlexNet):
    def __init__(self, 
                 num_classes: int = 200,
                 dropout: float = 0.5
        ):
        super().__init__(num_classes=num_classes, dropout=dropout)

        # [TODO] Modify feature extractor part in AlexNet
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            
            #### Add batch normalization ####
            nn.BatchNorm2d(64),
            
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            #### Add residual block ####
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, stride=2, downsample=nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(128))),
            ResidualBlock(128, 256, stride=2, downsample=nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(256))),
            ResidualBlock(256, 512, stride=2, downsample=nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(512))),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    
class MyAlexSE(AlexNet):
    def __init__(self, 
                 num_classes: int = 200,
                 dropout: float = 0.5
        ):
        super().__init__(num_classes=num_classes, dropout=dropout)

        # [TODO] Modify feature extractor part in AlexNet
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            
            #### Add SE block ####
            SEBlock(64),
            
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            
            #### Add SE block ####
            SEBlock(192),
            
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            #### Add SE block ####
            SEBlock(384),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            #### Add SE block ####
            SEBlock(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            #### Add SE block ####
            SEBlock(256),
            
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
                
class MyAlexTotal(AlexNet):
    def __init__(self, 
                 num_classes: int = 200,
                 dropout: float = 0.5
        ):
        super().__init__(num_classes=num_classes, dropout=dropout)

        # [TODO] Modify feature extractor part in AlexNet
        self.features = nn.Sequential(                  
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            
            #### Add batch normalization ####
            nn.BatchNorm2d(32),            
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            
            #### Add batch normalization ####
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            #### Add residual block ####
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, stride=2, downsample=nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(128))),
            nn.ReLU(inplace=False),
            
            #### Add inception block ####
            InceptionBlock(128),
            
            #### Add SE block ####
            SEBlock(256),            
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            #### Add residual block ####
            ResidualBlock(256, 256),
            ResidualBlock(256, 512, stride=2, downsample=nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(512))),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    