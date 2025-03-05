import argparse
import pathlib

import numpy as np
import timm
import torch
import torchvision.datasets as datasets
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader

from src.v_information import (
    extract_leaf_module_keys,
    extract_target_layer_features_yield,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--batch-size",
    "-b",
    type=int,
    default=32,
)
parser.add_argument(
    "--yield-every",
    "-y",
    type=int,
    default=100,
)
parser.add_argument(
    "--save-dir-path",
    "-s",
    type=pathlib.Path,
    default=pathlib.Path("outputs/penultimate_features"),
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
features_dict_generator = extract_target_layer_features_yield(
    model,
    dataloader,
    device,
    args.yield_every,
    target_layer_keys,
    pooling_mode="none",
)

# save extracted features.
args.save_dir_path.mkdir(parents=True, exist_ok=True)
print(f">> extracted layer features will be saved under `{args.save_dir_path}`.")
for i, partial_features_dict in enumerate(features_dict_generator):
    save_path = (
        args.save_dir_path
        / f"extracted_penultimate_features_batchsize:{args.batch_size}_{i:05d}.npz"
    )
    np.savez(save_path, **partial_features_dict)

print(">> all features are saved successfully.")
