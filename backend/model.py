import torch
from torch import nn
from torchvision import models
from torchvision.models import ResNet18_Weights

class PretrainedCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(PretrainedCNN, self).__init__()
        self.backbone = models.resnet18(weights = ResNet18_Weights.DEFAULT)  # Load pretrained weights
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)  # Replace final layer

    def forward(self, x):
        return self.backbone(x)

# model_path=path.join("tiny_vit_5m_224.dist_in22k")
vit5m=PretrainedCNN(10)
# vit5m.load_state_dict(torch.load("models/model.pth"))
# vit5m=create_model(model_path, pretrained=True)
# vit5m.load_state_dict(weights)
#what is the type of this model?
# <class 'timm.models.vision_transformer.VisionTransformer'>