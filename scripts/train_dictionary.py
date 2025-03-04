import argparse
import pathlib

import numpy as np

from src.dictionary_learning import train_dictionary

parser = argparse.ArgumentParser()
parser.add_argument(
    "--num-atom",
    "-n",
    type=int,
    default=10000,
)
parser.add_argument(
    "--batch-size",
    "-b",
    type=int,
    default=1024,
    help="Batch size for training the dictionary. If -1, use NMF instead of MiniBatchNMF.",
)
parser.add_argument(
    "--penultimate-features-path",
    "-i",
    type=pathlib.Path,
    default=pathlib.Path("outputs/extracted_penultimate_features.npz"),
)
parser.add_argument(
    "--save-path",
    "-o",
    type=pathlib.Path,
    default=pathlib.Path("outputs/dictionaries.npz"),
)
args = parser.parse_args()

# load penultimate features.
features_dict = np.load(args.penultimate_features_path)

print(">> loaded penultimate features:")
for k, v in features_dict.items():
    print(f"{k}: shape {v.shape}")

# reshape penultimate features.
penultimate_features = features_dict["layer4.2.act3"]
B, C, H, W = penultimate_features.shape
reshaped_penultimate_features = penultimate_features.transpose(0, 2, 3, 1).reshape(
    B * H * W, C
)

# train dictionary.
print(">> start dictionary learning:")
Z, D, error = train_dictionary(
    reshaped_penultimate_features, args.num_atom, batch_size=args.batch_size
)

print(">> result of dictionary learning:")
print(f"Z: shape {Z.shape}")
print(f"D: shape {D.shape}")
print(f"reconstruction error: {error}")

# save extracted features.
np.savez(args.save_path, Z=Z, D=D, error=error)
print(f">> Z and D are saved to `{args.save_path}`.")
