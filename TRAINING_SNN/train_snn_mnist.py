"""
snn_mnist.py

Energy-Aware SNN for MNIST classification, adapted from the architecture
proposed in:
    "Energy-Aware FPGA Implementation of Spiking Neural Network with
     LIF Neurons" (Ali, Navardi, Mohsenin - arXiv:2411.01628)

Architecture (adapted for MNIST):
    Input (784) -> FC1 (784->512) -> LIF hidden (512, w/ refractory)
                -> FC2 (512->10)  -> LIF output (10,  w/ refractory)

Key design choices mirrored from the paper:
    - Rate coding input encoding (Bernoulli spike generation per pixel)
    - 25 simulation time steps
    - 1st-order LIF neuron model (snntorch.Leaky)
    - Manual 5-timestep refractory period on both hidden and output layers
      (implemented explicitly rather than via any snnTorch built-in)
    - Loss computed and summed at every time step (cross-entropy on
      membrane potential / logits), then backpropagated once per batch

Libraries required: torch, torchvision, snntorch, numpy
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import snntorch as snn
from snntorch import spikegen


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

class Config:
    """Central place for all hyperparameters and run settings."""

    # Data
    data_dir = "./data"
    batch_size = 128

    # Encoding
    num_steps = 25          # number of simulation time steps (rate coding window)

    # Network architecture
    num_inputs = 784        # 28x28 flattened MNIST image
    num_hidden = 512        # hidden LIF layer size (per paper)
    num_outputs = 10        # digits 0-9

    # LIF neuron parameters (per paper / requirements)
    beta = 0.9
    threshold = 1.0
    # IMPORTANT: the actual "reset to zero on spike" behavior is implemented
    # manually inside RefractoryTracker.apply(), and only fires for neurons
    # that are actually allowed to spike (i.e. not in refractory). The
    # underlying snn.Leaky layers are therefore configured with
    # reset_mechanism="none" so snnTorch does NOT reset the membrane on its
    # own the instant threshold is crossed -- otherwise a neuron that is
    # currently blocked by refractory would still have its membrane wiped
    # to zero internally before our masking logic ever runs, which violates
    # "continue updating the membrane potential normally" during refractory.
    reset_mechanism = "none"
    learn_beta = False
    learn_threshold = False

    # Refractory period
    refractory_steps = 5    # neurons are silenced for this many steps after firing

    # Training
    num_epochs = 10
    learning_rate = 5e-4

    # Misc
    seed = 42
    model_save_path = "model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Dataset loading
# --------------------------------------------------------------------------- #

def get_dataloaders(config: Config):
    """
    Downloads MNIST (if needed), normalizes it, and returns train/test
    DataLoaders. Images are kept as [1, 28, 28] tensors here; flattening
    to 784 happens later in the forward pass so the raw dataset stays
    reusable / inspectable.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),                 # scales pixels to [0, 1]
        transforms.Normalize((0.1307,), (0.3081,))  # standard MNIST normalization
    ])

    train_dataset = datasets.MNIST(
        root=config.data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=config.data_dir, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, drop_last=True
    )

    return train_loader, test_loader


def encode_batch_to_spikes(images: torch.Tensor, num_steps: int) -> torch.Tensor:
    """
    Converts a batch of normalized MNIST images into rate-coded spike
    trains using snntorch.spikegen.rate().

    Args:
        images: tensor of shape [batch_size, 1, 28, 28] (normalized, ~[0,1]-ish
                after Normalize; rate coding expects values usable as spike
                probabilities, so we re-clamp to [0, 1] before encoding).
        num_steps: number of time steps to simulate (25 per paper).

    Returns:
        spike_data: tensor of shape [num_steps, batch_size, 784]
    """
    batch_size = images.shape[0]

    # Flatten each image to a 784-length vector.
    flat_images = images.view(batch_size, -1)

    # spikegen.rate expects pixel intensities in [0, 1] to use as
    # per-timestep Bernoulli firing probabilities (rate coding, per paper
    # Section 3.2). Normalization can push values slightly outside [0, 1],
    # so we clamp for a valid probability range.
    flat_images = torch.clamp(flat_images, 0.0, 1.0)

    # spikegen.rate returns shape [num_steps, batch_size, 784]
    spike_data = spikegen.rate(flat_images, num_steps=num_steps)

    return spike_data


# --------------------------------------------------------------------------- #
# Manual refractory mechanism
# --------------------------------------------------------------------------- #

class RefractoryTracker:
    """
    Tracks a per-neuron integer refractory counter for one LIF layer,
    implemented manually (not via any snnTorch built-in refractory logic).

    Usage per time step, per layer:
        1. decrement()          -- decrement all non-zero counters by 1
        2. apply(spk, mem)      -- suppress spikes for neurons still in
                                    their refractory window, reset counters
                                    for neurons that just fired, and zero
                                    the membrane potential of neurons that fired
    """

    def __init__(self, shape, refractory_steps: int, device: torch.device):
        self.refractory_steps = refractory_steps
        self.device = device
        self.counters = torch.zeros(shape, dtype=torch.int32, device=device)

    def reset(self, shape=None):
        """Reset all counters to zero (call at the start of every new batch)."""
        if shape is not None:
            self.counters = torch.zeros(shape, dtype=torch.int32, device=self.device)
        else:
            self.counters.zero_()

    def decrement(self):
        """Step 1: decrement all non-zero counters by one."""
        self.counters = torch.clamp(self.counters - 1, min=0)

    def apply(self, spk: torch.Tensor, mem: torch.Tensor):
        """
        Enforces the refractory rule on a raw spike/membrane pair produced
        by an snn.Leaky neuron for this time step.

        Rule (exactly as specified):
            - A neuron may spike only if its refractory counter is zero.
            - If a neuron spikes: emit the spike, zero its membrane
              potential, and set its counter to `refractory_steps`.
            - If the counter is > 0: suppress the spike output, but let the
              membrane potential continue updating normally (i.e. do NOT
              force it to zero just because it's in refractory).

        Args:
            spk: raw spike tensor from snn.Leaky for this step, shape [B, N]
            mem: membrane potential tensor for this step, shape [B, N]

        Returns:
            (masked_spk, masked_mem): spikes/membrane after refractory logic
        """
        # NOTE: because the LIF layers are configured with reset_mechanism=
        # "none", `mem` here has NOT already been reset by snnTorch, even if
        # `spk` is 1. That means membrane potential above threshold is
        # preserved for blocked neurons (correct "continue updating normally"
        # behavior), and the actual zero-reset below is the *only* place a
        # membrane reset happens, and only for neurons legitimately allowed
        # to fire this step.
        in_refractory = self.counters > 0          # [B, N] boolean mask

        # A neuron is only allowed to actually fire if it spiked AND is not
        # currently in its refractory window.
        allowed_spike = (spk.bool()) & (~in_refractory)

        # Suppress spikes for neurons still in refractory.
        masked_spk = torch.where(
            allowed_spike, spk, torch.zeros_like(spk)
        )

        # For neurons that just legitimately fired: reset membrane to zero
        # and set their refractory counter to refractory_steps.
        masked_mem = torch.where(
            allowed_spike, torch.zeros_like(mem), mem
        )

        # Update counters: neurons that fired get set to refractory_steps;
        # everyone else keeps whatever `decrement()` already left them at.
        self.counters = torch.where(
            allowed_spike,
            torch.full_like(self.counters, self.refractory_steps),
            self.counters
        )

        return masked_spk, masked_mem


# --------------------------------------------------------------------------- #
# Network definition
# --------------------------------------------------------------------------- #

class SNNMNIST(nn.Module):
    """
    Fully connected Spiking Neural Network for MNIST, following the
    784 -> 512 (LIF) -> 10 (LIF) architecture from the paper, with a
    manually implemented 5-step refractory period on both spiking layers.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Fully connected layers (no convolutions, per requirements).
        self.fc1 = nn.Linear(config.num_inputs, config.num_hidden)
        self.fc2 = nn.Linear(config.num_hidden, config.num_outputs)

        # Hidden LIF layer.
        self.lif1 = snn.Leaky(
            beta=config.beta,
            threshold=config.threshold,
            reset_mechanism=config.reset_mechanism,
            learn_beta=config.learn_beta,
            learn_threshold=config.learn_threshold,
        )

        # Output LIF layer.
        self.lif2 = snn.Leaky(
            beta=config.beta,
            threshold=config.threshold,
            reset_mechanism=config.reset_mechanism,
            learn_beta=config.learn_beta,
            learn_threshold=config.learn_threshold,
        )

    def forward(self, spike_input: torch.Tensor):
        """
        Runs the network over all simulation time steps.

        Args:
            spike_input: tensor of shape [num_steps, batch_size, 784]

        Returns:
            output_spikes: tensor of shape [num_steps, batch_size, num_outputs]
            output_membranes: tensor of shape [num_steps, batch_size, num_outputs]
        """
        num_steps, batch_size, _ = spike_input.shape
        device = spike_input.device

        # Initialize membrane potentials for both LIF layers.
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Ensure membrane tensors have the correct batch dimension
        # (snntorch initializes to a default shape; broadcast/expand as needed).
        mem1 = torch.zeros(batch_size, self.config.num_hidden, device=device)
        mem2 = torch.zeros(batch_size, self.config.num_outputs, device=device)

        # One refractory tracker per spiking layer, reset fresh every forward pass.
        refractory1 = RefractoryTracker(
            (batch_size, self.config.num_hidden), self.config.refractory_steps, device
        )
        refractory2 = RefractoryTracker(
            (batch_size, self.config.num_outputs), self.config.refractory_steps, device
        )

        output_spikes = []
        output_membranes = []

        for step in range(num_steps):
            # --- Step 1 of refractory rule: decrement all counters first ---
            refractory1.decrement()
            refractory2.decrement()

            # --- Read one spike frame for this time step ---
            x = spike_input[step]  # [batch_size, 784]

            # --- Hidden layer: FC1 -> LIF -> refractory logic ---
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1, mem1 = refractory1.apply(spk1, mem1)

            # --- Output layer: FC2 -> LIF -> refractory logic ---
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2, mem2 = refractory2.apply(spk2, mem2)

            output_spikes.append(spk2)
            output_membranes.append(mem2)

        output_spikes = torch.stack(output_spikes, dim=0)      # [T, B, 10]
        output_membranes = torch.stack(output_membranes, dim=0)  # [T, B, 10]

        return output_spikes, output_membranes


# --------------------------------------------------------------------------- #
# Training / evaluation
# --------------------------------------------------------------------------- #

def compute_batch_loss(output_membranes: torch.Tensor, targets: torch.Tensor,
                        loss_fn: nn.Module) -> torch.Tensor:
    """
    Computes cross-entropy loss at every time step (using the output layer's
    membrane potential as logits) and sums the losses across all time steps,
    as specified.

    Args:
        output_membranes: [num_steps, batch_size, num_outputs]
        targets: [batch_size] integer class labels
        loss_fn: nn.CrossEntropyLoss instance

    Returns:
        total_loss: scalar tensor, summed over all time steps
    """
    num_steps = output_membranes.shape[0]
    total_loss = 0.0
    for step in range(num_steps):
        total_loss = total_loss + loss_fn(output_membranes[step], targets)
    return total_loss


def batch_accuracy(output_spikes: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Computes classification accuracy for a batch by summing spike counts
    over all time steps per output neuron and taking the argmax (rate
    coding readout — the neuron that fired most often wins).

    Args:
        output_spikes: [num_steps, batch_size, num_outputs]
        targets: [batch_size]

    Returns:
        accuracy: float in [0, 1]
    """
    spike_counts = output_spikes.sum(dim=0)          # [batch_size, num_outputs]
    predictions = spike_counts.argmax(dim=1)         # [batch_size]
    correct = (predictions == targets).sum().item()
    return correct / targets.shape[0]


def train_one_epoch(model: SNNMNIST, train_loader: DataLoader, optimizer: torch.optim.Optimizer,
                     loss_fn: nn.Module, config: Config):
    """
    Runs one full training epoch. Returns average training loss and
    average training accuracy for the epoch.
    """
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0

    for images, labels in train_loader:
        images = images.to(config.device)
        labels = labels.to(config.device)

        # Rate-code the input images into spike trains.
        spike_input = encode_batch_to_spikes(images, config.num_steps)

        # Forward pass through the SNN.
        output_spikes, output_membranes = model(spike_input)

        # Loss: cross-entropy at every time step, summed over all steps.
        loss = compute_batch_loss(output_membranes, labels, loss_fn)

        # Backpropagation (once per batch, after summing per-step losses).
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_accuracy += batch_accuracy(output_spikes, labels)
        num_batches += 1

    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    return avg_loss, avg_accuracy


def evaluate(model: SNNMNIST, data_loader: DataLoader, config: Config):
    """
    Evaluates the model on a given DataLoader (no gradient updates).
    Returns average accuracy over the dataset.
    """
    model.eval()
    total_accuracy = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(config.device)
            labels = labels.to(config.device)

            spike_input = encode_batch_to_spikes(images, config.num_steps)
            output_spikes, _ = model(spike_input)

            total_accuracy += batch_accuracy(output_spikes, labels)
            num_batches += 1

    return total_accuracy / num_batches


def train_model(config: Config):
    """
    Top-level training routine: builds data loaders, model, optimizer,
    and loss function, then trains for config.num_epochs epochs, printing
    training loss/accuracy and test accuracy after every epoch. Saves the
    final model weights to config.model_save_path.
    """
    # Reproducibility.
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    print(f"Using device: {config.device}")

    train_loader, test_loader = get_dataloaders(config)

    model = SNNMNIST(config).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, config.num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, loss_fn, config
        )
        test_acc = evaluate(model, test_loader, config)

        print(
            f"Epoch [{epoch:02d}/{config.num_epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc * 100:.2f}% | "
            f"Test Acc: {test_acc * 100:.2f}%"
        )

    torch.save(model.state_dict(), config.model_save_path)
    print(f"Model weights saved to '{config.model_save_path}'")

    return model


# --------------------------------------------------------------------------- #
# Inference on a single sample (utility function)
# --------------------------------------------------------------------------- #

def predict_single_image(model: SNNMNIST, image: torch.Tensor, config: Config) -> int:
    """
    Runs inference on a single MNIST image (shape [1, 28, 28], already
    normalized) and returns the predicted digit class.
    """
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0).to(config.device)  # add batch dimension -> [1, 1, 28, 28]
        spike_input = encode_batch_to_spikes(image, config.num_steps)
        output_spikes, _ = model(spike_input)
        spike_counts = output_spikes.sum(dim=0)  # [1, num_outputs]
        prediction = spike_counts.argmax(dim=1).item()
    return prediction


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main():
    config = Config()
    train_model(config)


if __name__ == "__main__":
    main()
