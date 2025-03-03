import re
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader
from torchvision.models.feature_extraction import create_feature_extractor


def compute_v_information(feature: np.ndarray, z: np.ndarray) -> float:
    """Estimate the v-information for a specific atom by using an assumption that
    predictive family V as a class of linear probes with Gaussian prior.

    For details, please check "C Complexity measure" section of the original paper.

    Args:
        feature (np.ndarray): shape (B, C)

        z (np.ndarray): shape (B, 1)
            Coefficients of a specific atom.

    Returns:
        float: An estimated value of v-information.

    """
    assert feature.ndim == 2
    assert z.ndim == 1

    # specify ddof=1 to use unbiased variance.
    var_z = np.var(z, ddof=1)

    # estimate the coefficient of determination (R^2).
    model = LinearRegression().fit(feature, z)
    r2 = model.score(feature, z)

    return float((var_z * r2))


def compute_complexity_K(  # noqa: N802
    layer_outputs: list[np.ndarray],
    z: np.ndarray,
) -> float:
    """Estimate the complexity measure K for a specific atom.

    In the original paper, this is defined as K(z,x) in equation (2).

    Args:
        layer_outputs (list[np.ndarray]):
            A list of layer outputs. Each element must have shape (B, C).

        z (np.ndarray): shape (B, 1)
            Coefficients of a specific atom.


    Returns:
        float: An estimated value of complexity K.
            A higher value implies that the feature z is readily
            accessible and persists throughout the model layers.

    """
    v_informations = list()

    # TODO: use multiprocessing for parallel computation.
    for layer_output in layer_outputs:
        v_informations.append(compute_v_information(layer_output, z))

    return 1.0 - float(np.mean(v_informations))


def _extract_leaf_module_keys(
    module: nn.Module,
    prefix: str = "",
    target_key_patterns: Iterable[str] = [".*conv\\d$"],
) -> list[str]:
    """Recursively extract the keys of leaf modules that match any of the specified regex patterns.

    This function traverses the given module recursively and collects the full keys (names)
    of the leaf modules (i.e. modules with no children) whose names match at least one of the
    regex patterns provided in `target_key_patterns`. The `prefix` parameter is used to build
    the full module name during the recursion.

    Args:
        module (nn.Module): The PyTorch module to traverse.
        prefix (str, optional): The current prefix for module names. Defaults to "".
        target_key_patterns (List[str], optional): A list of regex patterns that the full module
            name must match.

    Returns:
        list[str]: A list of full module names (keys) that match at least one of the provided patterns.

    """
    # TODO: dynamically change target_key_patterns based on the model architecture.

    leaf_keys: list[str] = []
    for name, child in module.named_children():
        child_key = f"{prefix}.{name}" if prefix else name
        # if the child has no submodules (leaf module)
        if len(list(child.children())) == 0:
            if any(re.match(pattern, child_key) for pattern in target_key_patterns):
                leaf_keys.append(child_key)
        else:
            # recursively extend the list with keys from the child module.
            leaf_keys.extend(
                _extract_leaf_module_keys(child, child_key, target_key_patterns)
            )
    return leaf_keys


def extract_target_layer_features(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_global_average_pooling: bool = True,
) -> dict[str, np.ndarray]:
    """Extract features from specified target layers for the entire dataset using GPU batched inference.

    This function wraps the given model with a feature extractor to obtain outputs from target leaf modules.
    For each batch from the dataloader, the features are flattened (all dimensions except
    the batch dimension are collapsed) and stored. Finally, the features are concatenated
    along the batch dimension for each target layer and returned as a dictionary mapping layer
    keys to NumPy arrays.

    Args:
        model (nn.Module): The PyTorch model from which features will be extracted.
        dataloader (DataLoader): A DataLoader providing the dataset (e.g., the ImageNet dataset).
        device (torch.device): The device to run inference on.
        use_global_average_pooling (bool, optional): Whether to apply global average pooling
            to the output tensors. Defaults to True.

    Returns:
        dict[str, np.ndarray]: A dictionary where each key is a target layer name (str) and each
            value is a NumPy array of shape (N, D) containing the flattened features extracted from
            that layer across the entire dataset.
    """
    # Move the model to the target device and set it to evaluation mode.
    model.to(device)
    model.eval()

    # Extract target leaf module keys from the model.
    target_layer_keys = _extract_leaf_module_keys(model)

    # Create a feature extractor that returns outputs from the target layers.
    feature_extractor = create_feature_extractor(model, return_nodes=target_layer_keys)

    # Initialize a dictionary to collect features per target layer.
    features_by_layer: dict[str, list[np.ndarray]] = {
        key: [] for key in target_layer_keys
    }

    # Process the dataset in batches with no gradient computation.
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            # Obtain outputs as an OrderedDict from the feature extractor.
            batch_features = feature_extractor(images)
            for key, tensor in batch_features.items():
                # Apply global average pooling to the output tensor.
                if use_global_average_pooling:
                    reshaped_tensor = torch.mean(tensor, dim=(-1, -2))
                # Flatten the output tensor to 2D (batch_size, -1) and convert to a NumPy array.
                else:
                    reshaped_tensor = tensor.view(tensor.size(0), -1)

                features_by_layer[key].append(reshaped_tensor.cpu().numpy())

    # Concatenate batch outputs for each target layer along the batch dimension.
    concatenated_features: dict[str, np.ndarray] = {
        k: np.concatenate(v, axis=0) for k, v in features_by_layer.items()
    }

    return concatenated_features


if __name__ == "__main__":
    import pathlib
    from collections import defaultdict

    import timm
    import torchvision.datasets as datasets
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    # import torchvision.utils as vutils
    # from torch.utils.data import TensorDataset
    from torch.utils.data import DataLoader, Subset

    model = timm.create_model("resnet50", pretrained=True)

    # if you want to extract specific target layers, you can write like following.
    # this specifies 1,5,10,15,20,25,30,35,40,45,49-th layers of the ResNet50 model.
    # target_key_patterns = [
    #     "^conv1$",
    #     "^layer1.1.conv1$",
    #     "^layer1.2.conv3$",
    #     "^layer2.1.conv2$",
    #     "^layer2.3.conv1$",
    #     "^layer3.0.conv3$",
    #     "^layer3.2.conv2$",
    #     "^layer3.4.conv1$",
    #     "^layer3.5.conv3$",
    #     "^layer4.1.conv2$",
    #     "^layer4.2.conv3$",
    # ]

    # print(_extract_leaf_module_keys(model, target_key_patterns=target_key_patterns))

    # create a dummy dataset and dataloader for testing.
    # dummy_images = torch.randn(20, 3, 224, 224)
    # dummy_labels = torch.zeros(20)
    # dataset = TensorDataset(dummy_images, dummy_labels)
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    # create a dataset for the ImageNet validation set.
    config = resolve_data_config({}, model=model)
    val_transform = create_transform(**config)
    val_dataset = datasets.ImageFolder(
        root=pathlib.Path("data/imagenet/val"), transform=val_transform
    )

    # get indices information for each class.
    class_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(
        zip(val_dataset.imgs, val_dataset.targets, strict=True)
    ):
        class_to_indices[label].append(idx)

    # select 20 samples from each class and create a subset dataset.
    selected_indices = []
    for _, indices in class_to_indices.items():
        selected_indices.extend(indices[:20])

    subset_val_dataset = Subset(val_dataset, selected_indices)

    # create a dataloader for the subset dataset.
    val_loader = DataLoader(
        subset_val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )

    # save the first 3 batches of images for debugging.
    # for i, (images, labels) in enumerate(val_loader):
    #     print(images.shape, labels.shape)
    #     vutils.save_image(images, f"output_{i}.png", nrow=10, normalize=True)
    #     if i >= 2:
    #         break

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features_dict = extract_target_layer_features(model, val_loader, device)
    for k, v in features_dict.items():
        print(f"{k}: shape {v.shape}")
