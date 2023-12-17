import torch
import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
import timm


class ResNet50(nn.Module):
    def __init__(self, num_classes=1):
        super(ResNet50, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.resnet50(x)
    
    def get_name(self):
        return 'ResNet50'


class ResNet101(nn.Module):
    def __init__(self, num_classes=1):
        super(ResNet101, self).__init__()
        self.resnet101 = models.resnet101(pretrained=True)
        in_features = self.resnet101.fc.in_features
        self.resnet101.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.resnet101(x)
    
    def get_name(self):
        return 'ResNet101'
    
class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=1):
        super(EfficientNetB0, self).__init__()
        self.effnet = EfficientNet.from_pretrained('efficientnet-b0')
        in_features = self.effnet._fc.in_features
        self.effnet._fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.effnet(x)
    
    def get_name(self):
        return 'EfficientNetB0'

class EfficientNetB4(nn.Module):
    def __init__(self, num_classes=1):
        super(EfficientNetB4, self).__init__()
        self.effnet = EfficientNet.from_pretrained('efficientnet-b4')
        in_features = self.effnet._fc.in_features
        self.effnet._fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.effnet(x)
    
    def get_name(self):
        return 'EfficientNetB4'

    
class EfficientNetB7(nn.Module):
    def __init__(self, num_classes=1):
        super(EfficientNetB7, self).__init__()
        self.effnet = EfficientNet.from_pretrained('efficientnet-b7')
        in_features = self.effnet._fc.in_features
        self.effnet._fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.effnet(x)
    
    def get_name(self):
        return 'EfficientNetB7'
    
class VGG16(nn.Module):
    def __init__(self, num_classes=1):
        super(VGG16, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True)
        in_features = self.vgg16.classifier[6].in_features
        self.vgg16.classifier[6] = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.vgg16(x)
    
    def get_name(self):
        return 'VGG16'
    


class ViTaeV2(nn.Module):
    def __init__(self, num_classes=1):
        super(ViTaeV2, self).__init__()
        self.vit_model = timm.create_model('vit_base_patch16_384', pretrained=True)
        # Substitua 'vit_base_patch16_384' pelo nome do modelo Timm ViT que você quer usar

        # Ajuste a última camada para o número de classes desejado
        self.vit_model.head = nn.Linear(self.vit_model.head.in_features, num_classes)

    def forward(self, x):
        x = self.vit_model(x)
        return x
    
    def get_name(self):
        return 'ViTaeV2'