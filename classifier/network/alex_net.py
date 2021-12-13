from classifier import *
from acon import *
from typing import Tuple

'''
code from https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py
( same as: torchvision.models.alexnet() )

'''


class AlexNet(Module):
    def __init__(self, num_classes: int = 100, dropout: float = 0.5, sizes: Tuple = (64, 192, 384, 256, 256, 4096),
                 activation: str = 'relu') -> None:
        super().__init__()
        ac = [nn.ReLU(inplace=True) for _ in range(5)]
        if activation == 'acon':
            ac = [AconC(c) for c in sizes[:5]]
        elif activation == 'metaacon':
            ac = [MetaAconC(c) for c in sizes[:5]]

        self.features = nn.Sequential(
            nn.Conv2d(3, sizes[0], kernel_size=11, stride=4, padding=2),
            ac[0],
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(sizes[0], sizes[1], kernel_size=5, padding=2),
            ac[1],
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(sizes[1], sizes[2], kernel_size=3, padding=1),
            ac[2],
            nn.Conv2d(sizes[2], sizes[3], kernel_size=3, padding=1),
            ac[3],
            nn.Conv2d(sizes[3], sizes[4], kernel_size=3, padding=1),
            ac[4],
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(sizes[4] * 6 * 6, sizes[5]),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(sizes[5], sizes[5]),
            nn.ReLU(inplace=True),
            nn.Linear(sizes[5], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
