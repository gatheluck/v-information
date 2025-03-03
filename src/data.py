from collections import defaultdict

import torchvision.datasets as datasets
from torch.utils.data import Subset


def create_balanced_subset(
    dataset: datasets.ImageFolder, num_sample_per_class: int
) -> Subset:
    """Creates a balanced subset of the given ImageFolder dataset.

    This function extracts an equal number of samples from each class in the dataset.
    For each class, the first `num_sample_per_class` indices are selected.

    Args:
        dataset (datasets.ImageFolder): The dataset from which to create the subset.
        num_sample_per_class (int): The number of samples to extract from each class.

    Returns:
        Subset: A PyTorch Subset containing a balanced selection of samples from the dataset.

    """
    # get indices information for each class.
    class_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(zip(dataset.imgs, dataset.targets, strict=True)):
        class_to_indices[label].append(idx)

    # select samples from each class.
    selected_indices = []
    for _, indices in class_to_indices.items():
        selected_indices.extend(indices[:num_sample_per_class])

    return Subset(dataset, selected_indices)
