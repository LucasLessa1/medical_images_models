import torch
import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet


class ResNet50(nn.Module):
    """
    ResNet50 model for image classification.

    Args:
    - num_classes (int, optional): Number of output classes (default: 1).
    - input_channels (int, optional): Number of input channels (default: 3).

    Attributes:
    - resnet50 (torchvision.models.ResNet): ResNet50 backbone model.
    """

    def __init__(self,
                 num_classes: int = 1,
                 input_channels: int = 3
                 ) -> None:
        """
        Initialize ResNet50 model.

        Args:
        - num_classes (int, optional): Number of output classes
        (default: 1).
        - input_channels (int, optional): Number of input channels
        (default: 3).
        """
        super().__init__()
        self.resnet50 = models.resnet50(weights=True)

        # Modify the convolutional layer based on input channels
        self.resnet50.conv1 = nn.Conv2d(input_channels,
                                        64,
                                        kernel_size=7,
                                        stride=2,
                                        padding=3,
                                        bias=False)

        in_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes))

    def forward(self,
                input_image: torch.Tensor
                ) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
        - input_image (torch.Tensor): Input image tensor.

        Returns:
        - output (torch.Tensor): Output tensor from the model.
        """
        return self.resnet50(input_image)

    def get_name(self) -> str:
        """
        Get the name of the model.

        Returns:
        - name (str): Name of the model.
        """
        return 'ResNet50'


class ResNet101(nn.Module):
    """
    ResNet101 model for image classification.

    Args:
    - num_classes (int, optional): Number of output classes
    (default: 1).
    - input_channels (int, optional): Number of input channels
    (default: 3).

    Attributes:
    - resnet101 (torchvision.models.ResNet): ResNet101 backbone model.
    """

    def __init__(self,
                 num_classes: int = 1,
                 input_channels: int = 3
                 ) -> None:
        """
        Initialize ResNet101 model.

        Args:
        - num_classes (int, optional): Number of output classes
        (default: 1).
        - input_channels (int, optional): Number of input channels
        (default: 3).
        """
        super().__init__()
        self.resnet101 = models.resnet101(pretrained=True)
        self.resnet101.conv1 = nn.Conv2d(input_channels,
                                         64,
                                         kernel_size=7,
                                         stride=2,
                                         padding=3,
                                         bias=False)
        in_features = self.resnet101.fc.in_features
        self.resnet101.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
        - input_image (torch.Tensor): Input image tensor.

        Returns:
        - output (torch.Tensor): Output tensor from the model.
        """
        return self.resnet101(input_image)

    def get_name(self) -> str:
        """
        Get the name of the model.

        Returns:
        - name (str): Name of the model.
        """
        return 'ResNet101'


class EfficientNetB0(nn.Module):
    """
    EfficientNetB0 model for image classification.

    Args:
    - num_classes (int, optional): Number of output classes
    (default: 1).
    - input_channels (int, optional): Number of input channels
    (default: 3).

    Attributes:
    - effnet (efficientnet_pytorch.EfficientNet): EfficientNetB0 model.
    """

    def __init__(self, num_classes: int = 1, input_channels: int = 3) -> None:
        """
        Initialize EfficientNetB0 model.

        Args:
        - num_classes (int, optional): Number of output classes
        (default: 1).
        - input_channels (int, optional): Number of input channels
        (default: 3).
        """
        super().__init__()
        self.effnet = EfficientNet.from_pretrained('efficientnet-b0',
                                                   num_classes=num_classes,
                                                   in_channels=input_channels)

    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
        - input_image (torch.Tensor): Input image tensor.

        Returns:
        - output (torch.Tensor): Output tensor from the model.
        """
        return self.effnet(input_image)

    def get_name(self) -> str:
        """
        Get the name of the model.

        Returns:
        - name (str): Name of the model.
        """
        return 'EfficientNetB0'


class EfficientNetB4(nn.Module):
    """
    EfficientNetB4 model for image classification.

    Args:
    - num_classes (int, optional): Number of output classes
    (default: 1).
    - input_channels (int, optional): Number of input channels
    (default: 3).

    Attributes:
    - effnet (efficientnet_pytorch.EfficientNet): EfficientNetB4 model.
    """

    def __init__(self,
                 num_classes: int = 1,
                 input_channels: int = 3
                 ) -> None:
        """
        Initialize EfficientNetB4 model.

        Args:
        - num_classes (int, optional): Number of output classes
        (default: 1).
        - input_channels (int, optional): Number of input channels
        (default: 3).
        """
        super().__init__()
        self.effnet = EfficientNet.from_pretrained('efficientnet-b4',
                                                   num_classes=num_classes,
                                                   in_channels=input_channels)

    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
        - input_image (torch.Tensor): Input image tensor.

        Returns:
        - output (torch.Tensor): Output tensor from the model.
        """
        return self.effnet(input_image)

    def get_name(self) -> str:
        """
        Get the name of the model.

        Returns:
        - name (str): Name of the model.
        """
        return 'EfficientNetB4'


class EfficientNetB7(nn.Module):
    """
    EfficientNetB7 model for image classification.

    Args:
    - num_classes (int, optional): Number of output classes
    (default: 1).
    - input_channels (int, optional): Number of input channels
    (default: 3).

    Attributes:
    - effnet (efficientnet_pytorch.EfficientNet): EfficientNetB7 model.
    """

    def __init__(self,
                 num_classes: int = 1,
                 input_channels: int = 3
                 ) -> None:
        """
        Initialize EfficientNetB7 model.

        Args:
        - num_classes (int, optional): Number of output classes
        (default: 1).
        - input_channels (int, optional): Number of input channels
        (default: 3).
        """
        super().__init__()
        self.effnet = EfficientNet.from_pretrained('efficientnet-b7',
                                                   num_classes=num_classes,
                                                   in_channels=input_channels)

    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
        - input_image (torch.Tensor): Input image tensor.

        Returns:
        - output (torch.Tensor): Output tensor from the model.
        """
        return self.effnet(input_image)

    def get_name(self) -> str:
        """
        Get the name of the model.

        Returns:
        - name (str): Name of the model.
        """
        return 'EfficientNetB7'


class VGG16(nn.Module):
    """
    VGG16 model for image classification.

    Args:
    - num_classes (int, optional): Number of output classes
    (default: 1).
    - input_channels (int, optional): Number of input channels
    (default: 3).

    Attributes:
    - vgg16 (torchvision.models.VGG): VGG16 model.
    """

    def __init__(self,
                 num_classes: int = 1,
                 input_channels: int = 3
                 ) -> None:
        """
        Initialize VGG16 model.

        Args:
        - num_classes (int, optional): Number of output classes
          (default: 1).
        - input_channels (int, optional): Number of input channels
        (default: 3).
        """
        super().__init__()
        self.vgg16 = models.vgg16(pretrained=True)
        # Modify the first convolution layer
        self.vgg16.features[0] = nn.Conv2d(input_channels,
                                           64,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1)
        in_features = self.vgg16.classifier[6].in_features
        self.vgg16.classifier[6] = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
        - input_image (torch.Tensor): Input image tensor.

        Returns:
        - output (torch.Tensor): Output tensor from the model.
        """
        return self.vgg16(input_image)

    def get_name(self) -> str:
        """
        Get the name of the model.

        Returns:
        - name (str): Name of the model.
        """
        return 'VGG16'
