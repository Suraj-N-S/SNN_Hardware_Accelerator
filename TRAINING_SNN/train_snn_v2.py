"""
train_snn_paper.py

Implements the SNN exactly as described in:
  "Energy-Aware FPGA Implementation of Spiking Neural Network with LIF Neurons"
  Ali, Navardi, Mohsenin (2024)  arXiv:2411.01628

Key paper specs implemented here:
  1. RATE CODING  — pixel intensity = spike probability at each timestep
                    (Bernoulli draw per pixel per timestep, NOT static vector)
  2. LIF UPDATE   — U[t+1] = beta*U[t] + I[t+1], subtract reset
  3. ARCHITECTURE — Input linear (784→16) → Hidden LIF (16) → Output LIF (10)
  4. OPTIMIZER    — Adam, lr = 5e-4
  5. LOSS         — CrossEntropy summed across all timesteps
  6. TIMESTEPS    — 25
  7. BETA         — 0.875 (maps to  membrane - (membrane>>>3)  in lif.sv)

Differences from the paper (our constraints, not paper choices):
  - Paper uses 4096-input / 512-hidden for collision avoidance (64x64 images)
  - We use MNIST (28x28 = 784 pixels) with 16 hidden / 10 output
    to match lif.sv parameters (N_INPUTS, HIDDEN, OUTPUT)
  - Weight quantisation scale = 64 (signed 8-bit .mem files for BRAM)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ── Reproducibility ───────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ── Hyperparameters — MUST match lif.sv exactly ──────────────────────────────
INPUT_SIZE   = 784    # 28x28 MNIST pixels (flattened)
PROJ_SIZE    = 16     # linear projection -> matches N_INPUTS of hidden lif.sv
HIDDEN_SIZE  = 16     # 16 x lif.sv in hidden layer
OUTPUT_SIZE  = 10     # 10 x lif.sv in output layer (digits 0-9)
BETA         = 0.875  # decay factor -> lif.sv:  membrane - (membrane >>> 3)
THRESHOLD    = 1.0    # normalised threshold (= 100 in lif.sv after scaling)
TIMESTEPS    = 25     # paper Section 4.2.1: "model operates over 25 time steps"
BATCH_SIZE   = 128
NUM_EPOCHS   = 20     # more epochs since rate coding adds stochastic noise
WEIGHT_SCALE = 64     # float -> signed 8-bit for .mem files
DROPOUT_P    = 0.25   # paper Section 4.2.1: "dropout is applied for regularisation"
DEVICE       = torch.device("cpu")

# ── Dataset ───────────────────────────────────────────────────────────────────
# Paper: "normalizing the pixel values" -> ToTensor() gives [0,1] range
# We do NOT use Normalize() here because rate coding needs values in [0,1]
# so they can be used directly as Bernoulli probabilities
transform = transforms.ToTensor()   # output: [0.0, 1.0]

train_dataset = datasets.MNIST("./data", train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST("./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── Rate Coding ───────────────────────────────────────────────────────────────
# Paper Section 3.2:
#   "the normalized pixel value determines the probability of spike generation
#    at each time step ... a pixel value of 0.8 might mean there is an 80%
#    chance of a neuron firing at each time step"
#
# Implementation:
#   pixel_intensity in [0,1] -> torch.bernoulli(pixel_intensity) -> {0,1} spike
#   This is done fresh at EVERY timestep, so each call gives a different draw.
#   Over 25 timesteps, a pixel of intensity 0.8 will spike on average 20 times.

def rate_code(x_flat):
    """
    x_flat : [batch, 784]  float tensor in [0, 1]
    Returns : [batch, 784]  binary spike tensor  {0.0, 1.0}
    Called once per timestep — each call produces a new independent draw.
    """
    return torch.bernoulli(x_flat)

# ── Network Definition ────────────────────────────────────────────────────────
# Paper Figure 4 / Section 4.2.1:
#   Layer 1 (Input)  : linear transformation, flattens to pixel-count size
#   Layer 2 (Hidden) : LIF neurons, learnable threshold + beta, dropout
#   Layer 3 (Output) : linear + LIF neurons, 2 classes (we use 10 for MNIST)

class SNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Input projection: 784 (pixels) -> 16 (matches PROJ_SIZE / N_INPUTS of lif.sv)
        # Paper: "linear transformation that flattens the input images into a vector"
        self.input_proj = nn.Linear(INPUT_SIZE, PROJ_SIZE, bias=False)

        # Hidden layer: 16 -> 16  (16 instances of lif.sv)
        # Paper: "Leaky Integrate-and-Fire neurons, with learnable threshold and beta"
        # We keep beta fixed (must match hardware lif.sv exactly)
        self.fc1  = nn.Linear(PROJ_SIZE, HIDDEN_SIZE, bias=False)
        self.lif1 = snn.Leaky(
            beta=BETA,
            threshold=THRESHOLD,
            reset_mechanism="subtract",   # paper eq 4: U_rest subtraction on spike
            spike_grad=surrogate.fast_sigmoid()
        )

        # Paper: "dropout is applied for regularization"
        self.drop = nn.Dropout(p=DROPOUT_P)

        # Output layer: 16 -> 10  (10 instances of lif.sv)
        self.fc2  = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE, bias=False)
        self.lif2 = snn.Leaky(
            beta=BETA,
            threshold=THRESHOLD,
            reset_mechanism="subtract",
            spike_grad=surrogate.fast_sigmoid()
        )

    def forward(self, x_flat):
        """
        x_flat : [batch, 784]  pixel intensities in [0,1]
        Returns : spike_rec [TIMESTEPS, batch, 10],  mem_rec [TIMESTEPS, batch, 10]
        """
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spike_rec = []
        mem_rec   = []

        for t in range(TIMESTEPS):
            # ── Rate coding: new Bernoulli spike draw each timestep ──────────
            # Paper: "At each time step, a spike is generated with a probability
            #         corresponding to the pixel's normalized intensity"
            spk_in = rate_code(x_flat)          # [batch, 784]  binary

            # ── Input projection (binary spikes -> 16-dim current) ───────────
            projected = self.input_proj(spk_in) # [batch, 16]

            # ── Hidden LIF layer ─────────────────────────────────────────────
            cur1       = self.fc1(projected)    # weighted sum -> synaptic current
            spk1, mem1 = self.lif1(cur1, mem1)  # LIF dynamics + spike
            spk1       = self.drop(spk1)        # dropout for regularisation

            # ── Output LIF layer ─────────────────────────────────────────────
            cur2       = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spike_rec.append(spk2)
            mem_rec.append(mem2)

        # Stack along time dimension: [TIMESTEPS, batch, OUTPUT_SIZE]
        return torch.stack(spike_rec), torch.stack(mem_rec)

# ── Training ──────────────────────────────────────────────────────────────────
model = SNN().to(DEVICE)

# Paper Section 4.2.1: "Adam optimizer with a learning rate of 5e-4"
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

# Paper: "Cross-entropy loss is computed across all time steps, summing up
#         to form the total loss for each training instance"
loss_fn = nn.CrossEntropyLoss()

print("=" * 60)
print("SNN Training  —  paper: arXiv:2411.01628")
print("=" * 60)
print(f"  Coding       : Rate coding (Bernoulli, {TIMESTEPS} timesteps)")
print(f"  Architecture : {INPUT_SIZE} -> proj {PROJ_SIZE} -> LIF {HIDDEN_SIZE} -> LIF {OUTPUT_SIZE}")
print(f"  Beta         : {BETA}  (= membrane - membrane>>>3  in lif.sv)")
print(f"  Threshold    : {THRESHOLD}  (= 100 in lif.sv after weight scaling)")
print(f"  Optimizer    : Adam, lr=5e-4")
print(f"  Loss         : CrossEntropy summed over all {TIMESTEPS} timesteps")
print(f"  Epochs       : {NUM_EPOCHS}")
print("=" * 60 + "\n")

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for data, targets in train_loader:
        # Flatten 28x28 -> 784, keep in [0,1] for Bernoulli sampling
        data    = data.view(data.size(0), -1).to(DEVICE)   # [batch, 784]
        targets = targets.to(DEVICE)

        optimizer.zero_grad()

        spike_out, mem_out = model(data)
        # spike_out: [25, batch, 10]
        # mem_out:   [25, batch, 10]

        # Paper: "Cross-entropy loss computed across all time steps, summing up"
        # Sum membrane potentials over time -> use as logits for cross-entropy
        loss = loss_fn(mem_out.sum(0), targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Accuracy: class with most output spikes over all timesteps
        pred     = spike_out.sum(0).argmax(dim=1)
        correct += (pred == targets).sum().item()
        total   += targets.size(0)

    train_acc = 100.0 * correct / total
    avg_loss  = total_loss / len(train_loader)
    print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}%")

# ── Test Accuracy ─────────────────────────────────────────────────────────────
model.eval()
correct = total = 0

with torch.no_grad():
    for data, targets in test_loader:
        data    = data.view(data.size(0), -1).to(DEVICE)
        targets = targets.to(DEVICE)
        spike_out, _ = model(data)
        pred     = spike_out.sum(0).argmax(dim=1)
        correct += (pred == targets).sum().item()
        total   += targets.size(0)

test_acc = 100.0 * correct / total
print(f"\nTest Accuracy: {test_acc:.2f}%")

# ── Export weights to .mem files for BRAM loading in Verilog ─────────────────
# Paper Section 4.3: "weights ... represented using 16-bit signed fixed-point Q1.15"
# We use signed 8-bit (simpler for your lif.sv) with scale=64
# Format: one weight per line, 2-digit uppercase hex (two's complement)

os.makedirs("mem_files", exist_ok=True)

def export_weights(weight_tensor, filename, scale=WEIGHT_SCALE):
    """
    Quantise float weights to signed 8-bit integers.
    Save as 2-digit hex, one value per line (row-major order).
    Negative values stored as two's complement (e.g. -5 -> FB).
    """
    w = weight_tensor.detach().cpu().numpy()
    with open(filename, "w") as f:
        for row in w:
            for val in row:
                q = int(round(float(val) * scale))
                q = max(-128, min(127, q))     # clip to signed 8-bit range
                f.write(f"{q & 0xFF:02X}\n")   # two's complement hex
    total_weights = w.shape[0] * w.shape[1]
    print(f"  Saved: {filename}  ({w.shape[0]}x{w.shape[1]} = {total_weights} weights)")

print("\nExporting weights to .mem files...")
export_weights(model.input_proj.weight, "mem_files/weights_input_proj.mem")
export_weights(model.fc1.weight,        "mem_files/weights_layer1.mem")
export_weights(model.fc2.weight,        "mem_files/weights_layer2.mem")

# ── Export one test sample for tb_top.sv verification ────────────────────────
# Save the spike train for one image so Verilog testbench can replay it
print("\nExporting test sample for tb_top.sv...")
torch.manual_seed(42)   # fix seed so Python and tb_top.sv see same spike train

sample_img, sample_label = test_dataset[0]
sample_flat = sample_img.view(1, -1).to(DEVICE)   # [1, 784]

# Generate the 25-timestep spike train deterministically (seed fixed above)
sample_spikes = []   # list of 25 binary vectors, each of length 784
for t in range(TIMESTEPS):
    spk = rate_code(sample_flat).squeeze().cpu().numpy().astype(np.uint8)
    sample_spikes.append(spk)

# Project each spike frame through input_proj to get the 16-dim vector
model.eval()
with torch.no_grad():
    mem1 = model.lif1.init_leaky()
    mem2 = model.lif2.init_leaky()
    torch.manual_seed(42)   # reset seed for exact replay
    spike2_counts = torch.zeros(OUTPUT_SIZE)
    projected_frames = []
    for t in range(TIMESTEPS):
        spk_in    = rate_code(sample_flat)
        projected = model.input_proj(spk_in)
        projected_frames.append(projected.squeeze().cpu().numpy())
        spk1, mem1 = model.lif1(model.fc1(projected), mem1)
        spk2, mem2 = model.lif2(model.fc2(spk1), mem2)
        spike2_counts += spk2.squeeze().cpu()

predicted_class = int(spike2_counts.argmax().item())

# Save the 25 projected 16-dim frames (quantised) for tb_top.sv
with open("mem_files/test_sample.mem", "w") as f:
    f.write(f"// label={sample_label}  prediction={predicted_class}\n")
    for t, frame in enumerate(projected_frames):
        f.write(f"// timestep {t+1}\n")
        for val in frame:
            q = int(round(float(val) * WEIGHT_SCALE))
            q = max(-128, min(127, q))
            f.write(f"{q & 0xFF:02X}\n")
print(f"  Saved: mem_files/test_sample.mem")
print(f"  True label : {sample_label}")
print(f"  Predicted  : {predicted_class}  {'✓ CORRECT' if predicted_class == sample_label else '✗ WRONG'}")
print(f"\nSpike counts per output class: {spike2_counts.int().tolist()}")
print(f"Winning class (most spikes)  : {predicted_class}")

print("\n" + "=" * 60)
print("All .mem files saved in mem_files/")
print("Next step: write tb_top.sv using weights_layer1.mem + weights_layer2.mem")
print("=" * 60)
