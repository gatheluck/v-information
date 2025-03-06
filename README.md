# V-information based Complexity Measure

![python versions](https://img.shields.io/badge/python-3.12-blue)
[![tests](https://github.com/cvpaperchallenge/Ascender/actions/workflows/lint-and-test.yaml/badge.svg)](https://github.com/cvpaperchallenge/Ascender/actions/workflows/lint-and-test.yaml)
[![MIT License](https://img.shields.io/github/license/cvpaperchallenge/Ascender?color=green)](LICENSE)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Typing: mypy](https://img.shields.io/badge/typing-mypy-blue)](https://github.com/python/mypy)

This is an unofficial pytorch implementation of the experiments executed in the paper, "[Understanding Visual Feature Reliance through the Lens of Complexity](https://arxiv.org/abs/2407.06076)" [Fel+, NeurIPS2024].

> [!WARNING]
> This repo is still work in progress!

## What is this repo about?

In the original paper, a method is proposed for calculating a complexity mesure for the various concepts acquired by an ML model. Specifically, the complexity of each concept is determined by evaluating the layer depth at which the processing of information related to that concept is completed in the ML model. In this repo, we provide the code to compute the final complexity mesure $K(\bm{z}, \bm{x})$.

## Scripts

### 1. `scripts/extract_penultimate_features.py`

This code saves the feature of the penultimate layer $f_{n}(\bm{x})$ used in dictionary learning as an npz file. Following the original paper, we save the feature representations after applying ReLU.

### 2. `scripts/train_dictionary.py`

This code loads the feature of the penultimate layer and performs dictionary learning. NMF is used for the dictionary learning. For implementation details, please refer to the "A Feature Extraction" section in the Appendix of the original paper. The training results are saved as an npz file.
