from classifier import *
from acon import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import acon

class VGG16_Meta_ACON(Module):
    def __init__(self):
        super(VGG16_Meta_ACON, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            acon.MetaAconC(64),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            acon.MetaAconC(128),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            acon.MetaAconC(256),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            acon.MetaAconC(512),
            nn.MaxPool2d(2, 2),
        )

        self.dense = nn.Sequential(
            nn.Linear(512, 4096),
            acon.AconC_FC(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            acon.AconC_FC(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 100)

        )

    def forward(self, x):
        x = self.conv(x)
        x = Function.flatten(x)
        return self.dense(x)


class VGG16_MetaACON_Classifier(NNClassifier):
    def __init__(self, training_l: LabeledDataset, validation: LabeledDataset,
                 training_ul: Optional[UnlabeledDataset] = None,):
        """

        :param training_l:
        :param validation:
        :param training_ul:
        :param n_way: n ways of convolution layers
        :param depth: depth in each scale space. The length of the list must be 3
        """
        super(VGG16_MetaACON_Classifier, self).__init__(VGG16_Meta_ACON(), training_l, validation, training_ul)
        self.optim = SGD(self.network.parameters(), lr=1e-3, momentum=0.99)
        self.loss = CrossEntropyLoss()

    def predict(self, x: Tensor):
        return self._pred(x)

