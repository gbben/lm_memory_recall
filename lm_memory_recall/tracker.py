import torch
import torch.nn as nn
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import networkx as nx
from transformers import AutoTokenizer, AutoModelForCausalLM


class ActivationTracker:
    def __init__(self, model):
        self.model = model
        self.activations = defaultdict(list)
        self.hooks = []

    def _get_activation(self, name):
        def hook(model, input, output):
            self.activations[name].append(output.detach())

        return hook

    def attach_hooks(self):
        """Attach hooks to all layers we want to track"""
        for name, layer in self.model.named_modules():
            if isinstance(layer, (nn.Linear, nn.MultiheadAttention)):
                hook = layer.register_forward_hook(self._get_activation(name))
                self.hooks.append(hook)

    def clear_activations(self):
        """Clear stored activations"""
        self.activations = defaultdict(list)

    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class ActivationTrackerBatch:
    def __init__(self, model):
        self.model = model
        self.activations = defaultdict(list)
        self.hooks = []

    def _get_activation(self, name):
        def hook(model, input, output):
            # Handle both batched and unbatched outputs
            if isinstance(output, torch.Tensor):
                # If output is batched, store mean across batch dimension
                if output.dim() > 2:  # Batched
                    self.activations[name].append(output.mean(dim=0).detach())
                else:  # Unbatched
                    self.activations[name].append(output.detach())
            else:
                # For tuple outputs (like in attention layers)
                processed_output = []
                for tensor in output:
                    if tensor.dim() > 2:  # Batched
                        processed_output.append(tensor.mean(dim=0).detach())
                    else:  # Unbatched
                        processed_output.append(tensor.detach())
                self.activations[name].append(tuple(processed_output))

        return hook

    def attach_hooks(self):
        """Attach hooks to all layers we want to track"""
        for name, layer in self.model.named_modules():
            if isinstance(layer, (nn.Linear, nn.MultiheadAttention)):
                hook = layer.register_forward_hook(self._get_activation(name))
                self.hooks.append(hook)

    def clear_activations(self):
        """Clear stored activations"""
        self.activations = defaultdict(list)

    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class ActivationVisualizer:
    def __init__(self, model_architecture):
        """
        Initialize visualizer with model architecture information
        model_architecture: dict with layer sizes and connections
        """
        self.model_architecture = model_architecture

    def plot_heatmap(self, activations, title="Activation Heatmap"):
        """Plot activation intensity across layers"""
        plt.figure(figsize=(15, 8))
        sns.heatmap(activations, cmap="YlOrRd", xticklabels="auto", yticklabels="auto")
        plt.title(title)
        plt.xlabel("Neurons")
        plt.ylabel("Layers")
        plt.show()

    def plot_activation_flow(self, activations, threshold=0.5):
        """Plot network graph showing activation flow between layers"""
        G = nx.DiGraph()

        # Create nodes for each layer
        pos = {}
        for layer_idx, (layer_name, layer_size) in enumerate(
            self.model_architecture.items()
        ):
            for neuron in range(layer_size):
                node_id = f"{layer_name}_{neuron}"
                G.add_node(node_id)
                pos[node_id] = (layer_idx, neuron - layer_size / 2)

        # Add edges based on activation strength
        for layer_idx in range(len(self.model_architecture) - 1):
            current_layer = list(self.model_architecture.keys())[layer_idx]
            next_layer = list(self.model_architecture.keys())[layer_idx + 1]

            for i in range(self.model_architecture[current_layer]):
                for j in range(self.model_architecture[next_layer]):
                    if activations[layer_idx][i] > threshold:
                        G.add_edge(
                            f"{current_layer}_{i}",
                            f"{next_layer}_{j}",
                            weight=activations[layer_idx][i],
                        )

        plt.figure(figsize=(12, 8))
        nx.draw(
            G,
            pos,
            with_labels=False,
            node_color="lightblue",
            node_size=100,
            edge_color="gray",
            arrows=True,
            width=[G[u][v]["weight"] for u, v in G.edges()],
        )
        plt.title("Activation Flow Network")
        plt.show()

    def create_activation_animation(self, activation_sequence, interval=200):
        """Create animation of activation patterns over time"""
        fig, ax = plt.subplots(figsize=(12, 8))

        def update(frame):
            ax.clear()
            sns.heatmap(
                activation_sequence[frame],
                cmap="YlOrRd",
                ax=ax,
                xticklabels="auto",
                yticklabels="auto",
            )
            ax.set_title(f"Activation Pattern - Step {frame+1}")

        anim = FuncAnimation(
            fig, update, frames=len(activation_sequence), interval=interval, repeat=True
        )
        return anim


def run_memory_experiment(
    model, tracker, questions, control_questions, rest_tasks, block_size=3
):
    """
    Run experiment with memory questions, control questions, and rest periods

    Args:
        model: The language model to analyze
        tracker: ActivationTracker instance
        questions: List of memory-testing questions
        control_questions: List of control questions
        rest_tasks: List of rest period tasks
        block_size: Number of questions per block before rest period
    """
    results = {
        "memory_activations": defaultdict(list),
        "control_activations": defaultdict(list),
        "rest_activations": defaultdict(list),
        "temporal_sequence": [],  # Store sequence of activations for temporal analysis
    }

    def process_block(questions, block_type, rest_idx):
        for question in questions:
            tracker.clear_activations()
            _ = model(question)

            # Store activations for each layer
            for layer_name, activations in tracker.activations.items():
                results[f"{block_type}_activations"][layer_name].append(
                    torch.mean(torch.stack(activations), dim=0)
                )

            # Store temporal sequence
            results["temporal_sequence"].append(
                {
                    "type": block_type,
                    "activations": {
                        name: acts[-1] for name, acts in tracker.activations.items()
                    },
                }
            )

        # Add rest period after block
        tracker.clear_activations()
        rest_output = model(rest_tasks[rest_idx % len(rest_tasks)])

        # Store rest period activations
        for layer_name, activations in tracker.activations.items():
            results["rest_activations"][layer_name].append(
                torch.mean(torch.stack(activations), dim=0)
            )

        results["temporal_sequence"].append(
            {
                "type": "rest",
                "activations": {
                    name: acts[-1] for name, acts in tracker.activations.items()
                },
            }
        )

    # Process memory questions in blocks
    for i in range(0, len(questions), block_size):
        block = questions[i : i + block_size]
        process_block(block, "memory", i // block_size)

    # Process control questions in blocks
    for i in range(0, len(control_questions), block_size):
        block = control_questions[i : i + block_size]
        process_block(block, "control", i // block_size)

    return results


def run_memory_experiment_batched(
    model, tracker, questions, control_questions, rest_tasks, tokenizer, block_size=3
):
    """
    Run experiment with batched inputs

    Args:
        model: The language model to analyze
        tracker: ActivationTracker instance
        questions: List of memory-testing questions
        control_questions: List of control questions
        rest_tasks: List of rest tasks
        tokenizer: Tokenizer for processing inputs
        block_size: Number of questions per block before rest period
    """
    results = {
        "memory_activations": defaultdict(list),
        "control_activations": defaultdict(list),
        "rest_activations": defaultdict(list),
        "temporal_sequence": [],
    }

    def process_block(questions, block_type, rest_idx):
        for question in questions:
            tracker.clear_activations()

            # Format input as chat and tokenize
            messages = [{"role": "user", "content": question}]
            inputs = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )

            # Run model
            with torch.no_grad():
                model_output = model(inputs)

            # Store activations for each layer
            for layer_name, activations in tracker.activations.items():
                # Handle both batched and unbatched activations
                if isinstance(activations[-1], torch.Tensor):
                    results[f"{block_type}_activations"][layer_name].append(
                        activations[-1]  # Already averaged across batch in hook
                    )
                else:
                    # For tuple outputs, store each tensor separately
                    results[f"{block_type}_activations"][layer_name].append(
                        tuple(tensor for tensor in activations[-1])
                    )

            # Store temporal sequence
            results["temporal_sequence"].append(
                {
                    "type": block_type,
                    "activations": {
                        name: acts[-1] for name, acts in tracker.activations.items()
                    },
                }
            )

        # Add rest period after block
        tracker.clear_activations()
        messages = [{"role": "user", "content": rest_tasks[rest_idx % len(rest_tasks)]}]
        rest_inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )

        with torch.no_grad():
            rest_output = model(rest_inputs)

        # Store rest period activations
        for layer_name, activations in tracker.activations.items():
            if isinstance(activations[-1], torch.Tensor):
                results["rest_activations"][layer_name].append(activations[-1])
            else:
                results["rest_activations"][layer_name].append(
                    tuple(tensor for tensor in activations[-1])
                )

        results["temporal_sequence"].append(
            {
                "type": "rest",
                "activations": {
                    name: acts[-1] for name, acts in tracker.activations.items()
                },
            }
        )

    # Process questions in blocks
    for i in range(0, len(questions), block_size):
        block = questions[i : i + block_size]
        process_block(block, "memory", i // block_size)

    for i in range(0, len(control_questions), block_size):
        block = control_questions[i : i + block_size]
        process_block(block, "control", i // block_size)

    return results


def analyze_activations(results):
    """
    Analyze differences in activation patterns between memory, control, and rest periods
    """
    analysis = {}

    # Analyze each type of activation
    for activation_type in ["memory", "control", "rest"]:
        activations = results[f"{activation_type}_activations"]

        for layer_name in activations.keys():
            acts = torch.stack(activations[layer_name])

            analysis[f"{activation_type}_{layer_name}"] = {
                "mean_activation": torch.mean(acts, dim=0),
                "std_activation": torch.std(acts, dim=0),
                "peak_activation": torch.max(acts, dim=0).values,
            }

    # Compare activation patterns
    for layer_name in results["memory_activations"].keys():
        memory_acts = torch.stack(results["memory_activations"][layer_name])
        control_acts = torch.stack(results["control_activations"][layer_name])
        rest_acts = torch.stack(results["rest_activations"][layer_name])

        # Calculate differences
        analysis[f"diff_{layer_name}"] = {
            "memory_vs_control": torch.mean(memory_acts, dim=0)
            - torch.mean(control_acts, dim=0),
            "memory_vs_rest": torch.mean(memory_acts, dim=0)
            - torch.mean(rest_acts, dim=0),
            "control_vs_rest": torch.mean(control_acts, dim=0)
            - torch.mean(rest_acts, dim=0),
        }

    return analysis


def run_example():
    # * rest period tasks
    rest_tasks = [
        # Simple character counting
        "Count the number of 'a's in this sentence.",
        # Case checking
        "Are all words in this sentence capitalized?",
        # Simple pattern matching
        "Does this sentence contain any numbers?",
        # Basic syntax checking
        "Check if this sentence ends with a period.",
        # Word length tasks
        "Find the longest word in this sentence.",
    ]

    # * memory tasks
    memory_tasks = [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What is the chemical symbol for gold?",
    ]

    # * control tasks
    control_tasks = [
        "Is 5 greater than 3?",
        "Complete: day follows ___",
        "What sound does a cat make?",
    ]

    # Initialize tracker and visualizer

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

    tracker = ActivationTracker(model)
    visualizer = ActivationVisualizer(model)

    # Run experiment
    results = run_memory_experiment_batched(
        model, tracker, memory_tasks, control_tasks, rest_tasks, tokenizer, block_size=3
    )

    # # Run experiment with rest periods
    # results = run_memory_experiment(
    #     model, tracker, memory_tasks, control_tasks, rest_tasks, block_size=3
    # )

    analysis = analyze_activations(results)

    # Visualize results including rest periods
    # Plot activation patterns for all three types
    for activation_type in ["memory", "control", "rest"]:
        visualizer.plot_heatmap(
            analysis[f"{activation_type}_hidden_layer_1"]["mean_activation"],
            title=f"{activation_type.capitalize()} Task Activation Pattern",
        )

    # Create animation showing transition between tasks and rest periods
    activation_sequence = [
        frame["activations"]["hidden_layer_1"] for frame in results["temporal_sequence"]
    ]
    _ = visualizer.create_activation_animation(activation_sequence)
    plt.show()


if __name__ == "__main__":
    run_example()
