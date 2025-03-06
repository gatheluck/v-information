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
    default=16384,
    help="Batch size for training the dictionary. If -1, use NMF instead of MiniBatchNMF.",
)
parser.add_argument(
    "--penultimate-features-dir-path",
    "-i",
    type=pathlib.Path,
    default=pathlib.Path("outputs/penultimate_features"),
)
parser.add_argument(
    "--save-path",
    "-o",
    type=pathlib.Path,
    default=pathlib.Path("outputs/dictionaries.npz"),
)
args = parser.parse_args()

# load penultimate features.
target_paths = list(
    args.penultimate_features_dir_path.glob("extracted_penultimate_features_*.npz")
)

loaded_features = [np.load(f)["layer4.2.act3"] for f in target_paths]
penultimate_features = np.concatenate(loaded_features, axis=0)

print(">> loaded penultimate features:")
print(f"shape {penultimate_features.shape}")

# reshape penultimate features.
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

# calculate variance of each atom and take mean.
variances = np.var(Z, axis=1)
print(f"mean of var_z: {np.mean(variances)}")
