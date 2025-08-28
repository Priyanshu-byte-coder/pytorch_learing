import torch
import torchvision
from torch import nn
def create_effnetb2_model(num_classes:int=3,
                          seed:int=42):

    #pretrained weight(transfer the learnings)
    weights=torchvision.models.EfficientNet_B2_Weights.DEFAULT

    #get effnetb2 transofrmer
    transforms=weights.transforms()

    #setup pretrained model instance
    model=torchvision.models.efficientnet_b2(weights=weights)

    #freeze the base layer in the model(this wil stop all layers from training)
    for param in model.parameters():
        param.requires_grad=False

    #change classifier head with random seed for reproducibility
    torch.manual_seed(seed)
    model.classifier=nn.Sequential(
        nn.Dropout(p=0.3,inplace=True),
        nn.Linear(in_features=1408,out_features=num_classes)
    )
    return model,transforms
