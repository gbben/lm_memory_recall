"""First small step to analyse internals."""

import torch
import torch.nn as nn
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


class ActivationTracker:
    def __init__(self, model):
        self.model = model
        self.activations = defaultdict(list)
        self.hooks = []
        self._register_hooks()

    def _activation_hook(self, name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                # Store the full activation tensor
                output_np = output.detach().cpu().float().numpy()  # [B, N, D]
                self.activations[name].append(output_np)

        return hook

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.LayerNorm)):
                hook = module.register_forward_hook(self._activation_hook(name))
                self.hooks.append(hook)

    def clear_activations(self):
        self.activations.clear()

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()


def compute_activation_statistics(activation_maps):
    """
    Objective:
    I want to analyze which neurons (features) in each layer show the strongest
    activation patterns during fact recollection. For each position in the
    sequence and each feature, I want to understand how "important" that
    activation is relative to other activations.
    Input data shape:

    activation_maps: (N, P, F) where

    N = number of samples (queries)
    P = number of positions (19 in your case)
    F = number of features (9216 in your case)


    For each position and feature, I want to:

    Calculate how that activation ranks compared to all other features at that position
    Convert that rank to a percentile to make it easier to interpret (0-100%)

    Expected output shape:

    percentile_ranks: (N, P, F) with same shape as input
    Each value will be between 0-100, representing how strongly that feature
    activated compared to other features at that position

    Each sample's position will have its features ranked relative to
    each other, with 100 representing the most strongly activated
    feature and 0 representing the least activated feature at that
    position.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Compute activation statistics across features for each position

    Args:
        activation_maps: numpy array of shape (N, P, F)
            N = number of samples
            P = number of positions
            F = number of features

    Returns:
        Dictionary containing:
        - z_scores: standardized activation values
        - magnitudes: absolute activation values
        - percentile_ranks: percentile rank of each feature compared to others at that
            position

    """
    # Convert list of arrays into a single array, concatenating along batch dimension
    activation_maps = np.concatenate(
        activation_maps, axis=0
    )  # Now shape is (total_batches, P, F)

    # Compute z-scores across the feature dimension (axis=-1)
    mean = np.mean(activation_maps, axis=-1, keepdims=True)
    std = np.std(activation_maps, axis=-1, keepdims=True)
    z_scores = (activation_maps - mean) / (std + 1e-6)

    # Compute magnitude of activations
    magnitudes = np.abs(activation_maps)

    # Compute percentile ranks of absolute values
    percentile_ranks = np.zeros_like(activation_maps)
    N, P, F = activation_maps.shape

    # For each batch and position, rank the features
    for batch in range(activation_maps.shape[0]):
        for pos in range(activation_maps.shape[1]):
            abs_vals = magnitudes[batch, pos]  # Shape: (F,)
            ranks = stats.rankdata(abs_vals)  # Shape: (F,)
            percentile_ranks[batch, pos] = ranks / len(ranks) * 100

    return {
        "z_scores": z_scores,
        "magnitudes": magnitudes,
        "percentile_ranks": percentile_ranks,
    }


def analyze_recollection_patterns(model, queries, tokenizer, is_chat=True):
    """
    Analyze activation patterns during fact recollection
    """
    tracker = ActivationTracker(model)
    activation_patterns = defaultdict(list)

    model.eval()
    with torch.no_grad():
        for query in queries:
            if is_chat:
                print("We're in Chat mode")
                inputs = tokenizer.apply_chat_template(
                    query, add_generation_prompt=True, return_tensors="pt"
                )
                _ = model(inputs.to("mps"))
            else:
                inputs = tokenizer(query, return_tensors="pt")
                _ = model(**{k: v.to("mps") for k, v in inputs.items()})

            # Store activation patterns
            for name, activations in tracker.activations.items():
                if activations:
                    activation_patterns[name].extend(activations)

            tracker.clear_activations()

    # Compute statistics for each layer
    statistics_results = {}
    for name, patterns in activation_patterns.items():
        if patterns:
            statistics_results[name] = compute_activation_statistics(patterns)

    return statistics_results, activation_patterns


def plot_activation_analysis(
    statistics_results, layer_name=None, plot_type="percentile"
):
    """
    Visualize activation analysis results
    """
    if layer_name is None:
        layer_name = list(statistics_results.keys())[0]

    stats_data = statistics_results[layer_name]

    if plot_type == "percentile":
        data = stats_data["percentile_ranks"][0]  # Take first sample
        title = f"Activation Percentile Ranks: {layer_name}"
        cmap = "viridis"
    elif plot_type == "z_score":
        data = stats_data["z_scores"][0]  # Take first sample
        title = f"Activation Z-Scores: {layer_name}"
        cmap = "RdBu_r"
    else:  # magnitude
        data = stats_data["magnitudes"][0]  # Take first sample
        title = f"Activation Magnitudes: {layer_name}"
        cmap = "YlOrRd"

    plt.figure(figsize=(15, 8))
    sns.heatmap(data, cmap=cmap, center=(plot_type == "z_score"))
    plt.title(title)
    plt.xlabel("Feature Dimension")
    plt.ylabel("Position")
    plt.tight_layout()

    return plt.gcf()


def plot_top_activations(statistics_results, layer_name=None, top_k=10):
    """
    Plot positions and features with highest activation
    """
    if layer_name is None:
        layer_name = list(statistics_results.keys())[0]

    stats_data = statistics_results[layer_name]
    magnitudes = stats_data["magnitudes"][0]  # Take first sample

    # Find top k activations
    flat_indices = np.argpartition(magnitudes.flatten(), -top_k)[-top_k:]
    positions, features = np.unravel_index(flat_indices, magnitudes.shape)

    plt.figure(figsize=(10, 6))
    plt.scatter(
        features, positions, c=magnitudes[positions, features], cmap="YlOrRd", s=100
    )
    plt.colorbar(label="Activation Magnitude")
    plt.title(f"Top {top_k} Activations: {layer_name}")
    plt.xlabel("Feature Index")
    plt.ylabel("Position")
    plt.tight_layout()

    return plt.gcf()
