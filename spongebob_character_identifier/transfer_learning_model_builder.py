import torch
from torch import nn
from torchvision.models import efficientnet_b0, efficientnet_b2, EfficientNet_B0_Weights, EfficientNet_B2_Weights
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url

# Fix for wrong hash error from: https://github.com/pytorch/vision/issues/7744
def get_state_dict(self, *args, **kwargs):
    kwargs.pop("check_hash")
    return load_state_dict_from_url(self.url, *args, **kwargs)
WeightsEnum.get_state_dict = get_state_dict

# Create an EffNetB0 feature extractor
def create_effnetb0(out_features: int,
                    device: torch.device,
                    seed: int=42):
    # 1. Get the base model with pretrained weights and send to target device
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights).to(device)

    # 2. Freeze the base model layers
    for param in model.features.parameters():
        param.requires_grad = False

    # 3. Set the seeds
    torch.manual_seed(seed)

    # 4. Change the classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2,
                   inplace=True),
        nn.Linear(in_features=1280, 
                  out_features=out_features,
                  bias=True)
    ).to(device)

    # 5. Give the model a name
    model.name = "effnetb0"
    print(f"[INFO] Created new {model.name} model.")
    return model

# Create an EffNetB2 feature extractor
def create_effnetb2(out_features: int,
                    device: torch.device,
                    seed: int=42):
    # 1. Get the base model with pretrained weights and send to target device
    weights = EfficientNet_B2_Weights.DEFAULT
    model = efficientnet_b2(weights=weights).to(device)

    # 2. Freeze the base model layers
    for param in model.features.parameters():
        param.requires_grad = False

    # 3. Set the seeds
    torch.manual_seed(seed)

    # 4. Change the classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features=1408, out_features=out_features)
    ).to(device)

    # 5. Give the model a name
    model.name = "effnetb2"
    print(f"[INFO] Created new {model.name} model.")
    return model