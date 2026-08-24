import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.network = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.network(self.pool(value))


class EmotionModel(nn.Module):
    """EfficientNet-B0 + channel attention binary emotion classifier."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.features = backbone.features
        self.attention = ChannelAttention(1280)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1280, num_classes))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.attention(self.features(value))
        value = self.pool(value).flatten(1)
        return self.classifier(value)
