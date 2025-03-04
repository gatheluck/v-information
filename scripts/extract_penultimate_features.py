import argparse
import pathlib

import numpy as np
import timm
import torch
import torchvision.datasets as datasets
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader

from src.v_information import extract_leaf_module_keys, extract_target_layer_features

parser = argparse.ArgumentParser()
parser.add_argument(
    "--batch-size",
    "-b",
    type=int,
    default=32,
)
parser.add_argument(
    "--save-path",
    "-s",
    type=pathlib.Path,
    default=pathlib.Path("outputs/extracted_penultimate_features.npz"),
)
args = parser.parse_args()

# create a ResNet50 model.
model = timm.create_model("resnet50", pretrained=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create subset of ImageNet train dataset.
config = resolve_data_config({}, model=model)
transform = create_transform(**config)
dataset = datasets.ImageFolder(
    root=pathlib.Path("data/imagenet/train"), transform=transform
)
dataloader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

# Move the model to the target device and set it to evaluation mode.
model.to(device)
model.eval()

# extract specific target layers. 49-th layers of the ResNet50 model.
target_key_patterns = ["^layer4.2.act3$"]
target_layer_keys = extract_leaf_module_keys(
    model, target_key_patterns=target_key_patterns
)

# extract penultimate features.
features_dict = extract_target_layer_features(
    model, dataloader, device, target_layer_keys, pooling_mode="none"
)

print(">> penultimate features:")
for k, v in features_dict.items():
    print(f"{k}: shape {v.shape}")

# save extracted features.
np.savez(args.save_path, **features_dict)
print(f">> extracted layer features are saved to `{args.save_path}`.")
