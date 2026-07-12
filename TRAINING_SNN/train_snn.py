# ─────────────────────────────────────────────────────────────
# Phase 3 — SNN Training Script
# Trains a 2-layer SNN on MNIST using snnTorch
# Exports weights as .mem files for BRAM loading in Verilog
# ─────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import os

# ── Reproducibility ───────────────────────────────────────────
torch.manual_seed(42)

# ── Network parameters ────────────────────────────────────────
# Must match your Verilog parameters exactly
INPUT_SIZE   = 16    # HDC hypervector dimension D
HIDDEN_SIZE  = 16    # number of LIF neurons in hidden layer
OUTPUT_SIZE  = 10    # one neuron per digit class (0-9)
BETA         = 0.875 # leak factor — matches >>> in Verilog (1 - 1/8)
THRESHOLD    = 1.0   # normalised threshold (will be scaled to 100 in Verilog)
NUM_STEPS    = 25    # number of timesteps per inference window
BATCH_SIZE   = 128
NUM_EPOCHS   = 10
WEIGHT_SCALE = 100   # multiply float weights by this before saving to .mem

# ── Dataset ───────────────────────────────────────────────────
# MNIST images are 28x28 = 784 pixels
# We flatten and reduce to INPUT_SIZE=16 via a linear projection
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True,  download=True, transform=transform)
test_dataset  = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

# ── Network definition ────────────────────────────────────────
class SNN(nn.Module):
    def __init__(self):
        super(SNN, self).__init__()

        # input projection: 784 pixels → 16 (HDC dimension)
        self.input_proj = nn.Linear(784, INPUT_SIZE, bias=False)

        # layer 1: 16 → 16 (hidden LIF layer)
        self.fc1  = nn.Linear(INPUT_SIZE, HIDDEN_SIZE, bias=False)
        self.lif1 = snn.Leaky(beta=BETA, threshold=THRESHOLD,
                               reset_mechanism='subtract')

        # layer 2: 16 → 10 (output LIF layer — no reset, pure integrator)
        self.fc2  = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE, bias=False)
        self.lif2 = snn.Leaky(beta=BETA, threshold=THRESHOLD,
                               reset_mechanism='subtract')

    def forward(self, x):
        # x shape: [batch, 784]

        # project input to HDC dimension
        x = self.input_proj(x)        # [batch, 16]

        # initialise membrane potentials to zero
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spike2_rec  = []   # record output spikes over all timesteps
        mem2_rec    = []   # record output membrane over all timesteps

        # run for NUM_STEPS timesteps
        for t in range(NUM_STEPS):
            # hidden layer
            cur1       = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)

            # output layer
            cur2       = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spike2_rec.append(spk2)
            mem2_rec.append(mem2)

        # stack over time: [NUM_STEPS, batch, OUTPUT_SIZE]
        return torch.stack(spike2_rec), torch.stack(mem2_rec)

# ── Training ──────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model     = SNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn   = nn.CrossEntropyLoss()

print("\nTraining SNN...")
print(f"  Input size:   {INPUT_SIZE}")
print(f"  Hidden size:  {HIDDEN_SIZE}")
print(f"  Output size:  {OUTPUT_SIZE}")
print(f"  Beta (leak):  {BETA}")
print(f"  Timesteps:    {NUM_STEPS}")
print(f"  Epochs:       {NUM_EPOCHS}\n")

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    correct    = 0
    total      = 0

    for batch_idx, (data, targets) in enumerate(train_loader):
        data    = data.view(data.size(0), -1).to(device)  # flatten
        targets = targets.to(device)

        optimizer.zero_grad()

        spike_out, mem_out = model(data)

        # loss on sum of output membrane potentials over time
        loss = loss_fn(mem_out.sum(0), targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # accuracy: class with most spikes
        pred     = spike_out.sum(0).argmax(dim=1)
        correct += (pred == targets).sum().item()
        total   += targets.size(0)

    train_acc = 100 * correct / total
    print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
          f"Loss: {total_loss/len(train_loader):.4f} | "
          f"Train Acc: {train_acc:.2f}%")

# ── Test accuracy ─────────────────────────────────────────────
model.eval()
correct = 0
total   = 0

with torch.no_grad():
    for data, targets in test_loader:
        data    = data.view(data.size(0), -1).to(device)
        targets = targets.to(device)

        spike_out, mem_out = model(data)
        pred     = spike_out.sum(0).argmax(dim=1)
        correct += (pred == targets).sum().item()
        total   += targets.size(0)

test_acc = 100 * correct / total
print(f"\nTest Accuracy: {test_acc:.2f}%")

# ── Export weights to .mem files ──────────────────────────────
# .mem files are loaded into BRAM using $readmemh in Verilog
# Format: one weight per line in hexadecimal
# Weights are scaled by WEIGHT_SCALE and clamped to 8-bit signed (-128 to 127)

os.makedirs('mem_files', exist_ok=True)

def export_weights(weight_tensor, filename, scale=WEIGHT_SCALE):
    weights = weight_tensor.detach().cpu().numpy()
    weights_scaled = np.clip(np.round(weights * scale), -128, 127).astype(np.int8)

    with open(filename, 'w') as f:
        for row in weights_scaled:
            for val in row:
                # write as 2-digit hex (two's complement for negative)
                f.write(f"{int(val) & 0xFF:02X}\n")

    print(f"  Saved: {filename}  "
          f"shape={weights_scaled.shape}  "
          f"min={weights_scaled.min()}  max={weights_scaled.max()}")
    return weights_scaled

print("\nExporting weights to .mem files...")
w1 = export_weights(model.fc1.weight, 'mem_files/weights_layer1.mem')
w2 = export_weights(model.fc2.weight, 'mem_files/weights_layer2.mem')
wi = export_weights(model.input_proj.weight, 'mem_files/weights_input_proj.mem')

# ── Export one test sample for Verilog testbench ──────────────
# Save one MNIST image encoded through input_proj as a .mem file
# This lets your Verilog testbench load a real input
print("\nExporting one test sample...")
model.eval()
with torch.no_grad():
    sample_data, sample_label = test_dataset[0]
    sample_flat  = sample_data.view(1, -1).to(device)
    sample_proj  = model.input_proj(sample_flat).squeeze().cpu().numpy()

    # quantise to integers
    sample_scaled = np.clip(np.round(sample_proj * WEIGHT_SCALE),
                            -128, 127).astype(np.int8)

    with open('mem_files/test_sample.mem', 'w') as f:
        for val in sample_scaled:
            f.write(f"{int(val) & 0xFF:02X}\n")

    print(f"  Saved: mem_files/test_sample.mem")
    print(f"  True label: {sample_label}")
    print(f"  Projected values: {sample_scaled}")

print("\nAll .mem files saved in mem_files/ folder")
print("\nNext step:")
print("  Copy mem_files/ folder into your Vivado project directory")
print("  These files will be loaded into BRAM using $readmemh")
