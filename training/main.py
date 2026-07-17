

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import snntorch as snn
from snntorch import functional as SF
import numpy as np
import os

# ── Parameters (match these to your Verilog parameters) ──────────────────────
BETA        = 0.5     # leak factor — same as α in your LIF (>>>1 = 0.5)
N_STEPS     = 25      # number of timesteps per sample (spike window)
BITWIDTH    = 16      # must match BITWIDTH in lif.v
BATCH_SIZE  = 128
EPOCHS      = 5
LR          = 1e-3
HIDDEN      = 128     # neurons in hidden layer
OUTPUT      = 10      # digits 0-9

# ── Dataset ───────────────────────────────────────────────────────────────────
transform = transforms.Compose([transforms.ToTensor()])

train_data = datasets.MNIST(root="data", train=True,  download=True, transform=transform)
test_data  = datasets.MNIST(root="data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False)

# ── Model ─────────────────────────────────────────────────────────────────────
# Two linear layers with LIF neurons after each
net = nn.Sequential(
    nn.Linear(784, HIDDEN, bias=False),   # W1: weight lookup when spike=1
    snn.Leaky(beta=BETA, init_hidden=True),
    nn.Linear(HIDDEN, OUTPUT, bias=False), # W2
    snn.Leaky(beta=BETA, init_hidden=True, output=True)
)

optimizer = torch.optim.Adam(net.parameters(), lr=LR)
loss_fn   = SF.ce_rate_loss()   # cross-entropy on spike rate

# Rate coding method
def rate_encode(imgs, n_steps):
    """
    imgs: [batch, 784] float tensor, values 0.0 to 1.0
    Returns: [n_steps, batch, 784] binary spike tensor
    """
    spikes = []
    for _ in range(n_steps):
        # each pixel fires with probability = pixel intensity
        spike = torch.bernoulli(imgs)   # 1 with prob=pixel, 0 otherwise
        spikes.append(spike)
    return torch.stack(spikes, dim=0)  # [T, B, 784]


# ── Training ──────────────────────────────────────────────────────────────────
def train_one_epoch(loader):
    net.train()
    total_loss = 0
    for imgs, labels in loader:
        imgs = imgs.view(imgs.size(0), -1)  # [B, 784]

        # reset membrane potentials
        for m in net.modules():
            if hasattr(m, 'reset_hidden'):
                m.reset_hidden()

        # rate encode: pixels → spike trains
        spike_input = rate_encode(imgs, N_STEPS)  # [T, B, 784]

        spk_rec = []
        for t in range(N_STEPS):
            spk_out, _ = net(spike_input[t])   # feed one timestep of spikes
            spk_rec.append(spk_out)

        spk_rec = torch.stack(spk_rec, dim=0)
        loss = loss_fn(spk_rec, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate(loader):
    net.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.view(imgs.size(0), -1)
            for m in net.modules():
              if hasattr(m, 'reset_hidden'):
                m.reset_hidden()
            spk_rec = []
            for _ in range(N_STEPS):
                spk_out, _ = net(imgs)
                spk_rec.append(spk_out)
            spk_rec  = torch.stack(spk_rec, dim=0)  # [T, B, 10]
            pred     = spk_rec.sum(0).argmax(1)      # most spikes = prediction
            correct += (pred == labels).sum().item()
            total   += labels.size(0)
    return 100 * correct / total

print("Training SNN on MNIST...")
for epoch in range(EPOCHS):
    loss = train_one_epoch(train_loader)
    acc  = evaluate(test_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}  loss={loss:.4f}  test_acc={acc:.2f}%")

# ── Weight extraction ─────────────────────────────────────────────────────────
# Get the two weight matrices
layers = [m for m in net.modules() if isinstance(m, nn.Linear)]
W1 = layers[0].weight.detach().numpy()   # shape: (128, 784)
W2 = layers[1].weight.detach().numpy()   # shape: (10, 128)

print(f"\nW1 shape: {W1.shape}  min={W1.min():.3f}  max={W1.max():.3f}")
print(f"W2 shape: {W2.shape}  min={W2.min():.3f}  max={W2.max():.3f}")

# ── Quantization: float → signed 16-bit fixed point ──────────────────────────
# Scale so that the largest weight fills ~half the signed range (leave headroom)
SCALE = (2**(BITWIDTH-1) - 1) / (2 * max(np.abs(W1).max(), np.abs(W2).max()))

W1_q = np.clip(np.round(W1 * SCALE), -(2**(BITWIDTH-1)), (2**(BITWIDTH-1))-1).astype(np.int16)
W2_q = np.clip(np.round(W2 * SCALE), -(2**(BITWIDTH-1)), (2**(BITWIDTH-1))-1).astype(np.int16)

print(f"\nQuantization scale: {SCALE:.4f}")
print(f"W1_q  min={W1_q.min()}  max={W1_q.max()}")
print(f"W2_q  min={W2_q.min()}  max={W2_q.max()}")

# ── Save as .npy (Python golden model uses this) ──────────────────────────────
os.makedirs("weights", exist_ok=True)
np.save("weights/W1_q.npy", W1_q)
np.save("weights/W2_q.npy", W2_q)
print("\nSaved: weights/W1_q.npy  weights/W2_q.npy")

# ── Save as .mem (Verilog $readmemh loads this into BRAM) ─────────────────────
def save_mem(weights, filename):
    """Save weight matrix as hex .mem file, row by row."""
    with open(filename, "w") as f:
        for row in weights:
            for val in row:
                # convert signed int16 to unsigned hex (two's complement)
                hex_val = format(int(val) & 0xFFFF, "04X")
                f.write(hex_val + "\n")
    print(f"Saved: {filename}")

save_mem(W1_q, "weights/W1.mem")
save_mem(W2_q, "weights/W2.mem")

print("\nDone. In your Verilog testbench, load weights like this:")
print('  $readmemh("weights/W1.mem", weight_bram_layer1);')
print('  $readmemh("weights/W2.mem", weight_bram_layer2);')
