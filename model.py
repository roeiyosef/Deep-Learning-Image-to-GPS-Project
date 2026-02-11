
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class VPSModel(nn.Module):
    def __init__(self, num_classes=300):
        super().__init__()

        res50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.feature_dim = res50.fc.in_features
        dropout_p = 0.1


        self.backbone = nn.Sequential(
            res50.conv1,
            res50.bn1,
            res50.relu,
            res50.maxpool,
            res50.layer1,
            nn.Dropout2d(p=dropout_p),
            res50.layer2,
            nn.Dropout2d(p=dropout_p),
            res50.layer3,
            nn.Dropout2d(p=dropout_p),
            res50.layer4,
            res50.avgpool,
            nn.Flatten()
        )

        self.regressor = nn.Sequential(
           nn.Linear(self.feature_dim, 1024),
           nn.BatchNorm1d(1024),
           nn.ReLU(),
           nn.Dropout(0.3),
           nn.Linear(1024, 256),
           nn.ReLU(),
           nn.Linear(256, 2)
        )


        self.embedding_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
           nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        pred_gps = self.regressor(features)
        emb = self.embedding_head(features)
        emb = F.normalize(emb, p=2, dim=1)
        pred_cls = self.classifier(features)
        return pred_gps, emb, pred_cls