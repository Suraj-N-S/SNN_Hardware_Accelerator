# ─────────────────────────────────────────────────────────────
# train_snn_paper.py  v3
# Follows arXiv:2411.01628
#
# Fixes applied vs v2:
#   - Fixed beta=0.9 (not learnable) — prevents divergence
#   - Fixed threshold=1.0 (not learnable) — stable training
#   - Timesteps increased 25→50 for MNIST sparsity
#   - Surrogate gradient handled by snntorch default
#   - Learning rate scheduler added for stable convergence
#
# Paper specs followed:
#   LIF Equation   : U[t+1] = βU[t] + I[t+1] - U_rest  (Eq.4)
#   Reset          : U[t+1] = 0                          (Eq.2)
#   Rate coding    : pixel intensity = spike probability  (Sec 3.2)
#   Optimizer      : Adam lr=5e-4                        (Sec 4.2.1)
#   Loss           : CrossEntropy across timesteps        (Sec 4.2.1)
#   Dropout        : 0.25                                (Sec 4.2.1)
#   Weight format  : Q1.15 16-bit                        (Sec 4.3)
#   Cascaded adder : binary inputs, no multipliers       (Sec 4.3)
# ─────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import spikegen
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import os

torch.manual_seed(42)
np.random.seed(42)

# ── Parameters ────────────────────────────────────────────────
INPUT_SIZE   = 784     # 28×28 pixels
HIDDEN_SIZE  = 128     # scaled from paper's 512
OUTPUT_SIZE  = 10      # MNIST digits
BETA         = 0.9     # fixed — paper trains this but we fix for stability
                       # hardware: membrane - (membrane >>> 4) ≈ 0.9375
THRESHOLD    = 1.0     # fixed — stable for hardware
NUM_STEPS    = 50      # increased from 25 — MNIST needs more steps for
                       # sufficient spike density with rate coding
BATCH_SIZE   = 128
NUM_EPOCHS   = 20
DROPOUT_P    = 0.25    # paper Section 4.2.1
WEIGHT_SCALE = 32768   # Q1.15: 2^15

# ── Dataset ───────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor()   # normalises to [0,1] for rate coding
])

train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True,  download=True, transform=transform)
test_dataset  = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0)

# ── Network ───────────────────────────────────────────────────
class SNN_Paper(nn.Module):
    def __init__(self):
        super(SNN_Paper, self).__init__()

        # paper Section 4.2.1 point 1:
        # linear transformation — 784 pixels → 128 hidden neurons
        # in hardware this is the cascaded adder (Section 4.3)
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE, bias=False)

        # paper Section 4.2.1 point 2:
        # LIF hidden layer — fixed beta and threshold for hardware stability
        # paper Equation 4: U[t+1] = βU[t] + I[t+1] - U_rest
        # paper Equation 2: reset to zero on spike
        self.lif1    = snn.Leaky(beta=BETA,
                                  threshold=THRESHOLD,
                                  reset_mechanism='zero',
                                  learn_beta=False,
                                  learn_threshold=False)
        self.dropout = nn.Dropout(p=DROPOUT_P)

        # paper Section 4.2.1 point 3:
        # linear transformation + LIF output layer
        self.fc2  = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE, bias=False)
        self.lif2 = snn.Leaky(beta=BETA,
                               threshold=THRESHOLD,
                               reset_mechanism='zero',
                               learn_beta=False,
                               learn_threshold=False)

    def forward(self, spike_data):
        """
        spike_data: [NUM_STEPS, batch, INPUT_SIZE] binary spikes
        paper Section 3.2 — rate coded from pixel intensities
        """
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spike2_rec = []
        mem2_rec   = []

        for t in range(NUM_STEPS):
            # paper Section 4.3: cascaded adder
            # inputs are binary → fc1 = conditional weight addition
            cur1       = self.fc1(spike_data[t])
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1       = self.dropout(spk1)
            cur2       = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spike2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spike2_rec), torch.stack(mem2_rec)

# ── Training ──────────────────────────────────────────────────
device    = torch.device('cpu')
model     = SNN_Paper().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=10, gamma=0.5)  # halve lr every 10 epochs
loss_fn   = nn.CrossEntropyLoss()

print("=" * 60)
print("SNN Training — arXiv:2411.01628  (v3 fixed)")
print("=" * 60)
print(f"  Input neurons  : {INPUT_SIZE} (rate coded)")
print(f"  Hidden neurons : {HIDDEN_SIZE}")
print(f"  Output neurons : {OUTPUT_SIZE}")
print(f"  Beta (fixed)   : {BETA} → Verilog: membrane-(membrane>>>4)")
print(f"  Threshold      : {THRESHOLD} → Verilog: {int(THRESHOLD*WEIGHT_SCALE)}")
print(f"  Reset          : zero (paper Equation 2)")
print(f"  Timesteps      : {NUM_STEPS}")
print(f"  Dropout        : {DROPOUT_P}")
print(f"  Optimizer      : Adam lr=5e-4 + StepLR scheduler")
print(f"  Loss           : CrossEntropy across timesteps")
print(f"  Weight format  : Q1.15 16-bit")
print()

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    correct    = 0
    total      = 0

    for data, targets in train_loader:
        data = data.view(data.size(0), -1).to(device)

        # rate coding — paper Section 3.2
        # pixel value = probability of spike at each timestep
        spike_data = spikegen.rate(data, num_steps=NUM_STEPS)

        targets = targets.to(device)
        optimizer.zero_grad()

        spike_out, mem_out = model(spike_data)

        # paper Section 4.2.1: cross-entropy across all timesteps
        loss = loss_fn(mem_out.sum(0), targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred        = spike_out.sum(0).argmax(dim=1)
        correct    += (pred == targets).sum().item()
        total      += targets.size(0)

    scheduler.step()
    train_acc = 100 * correct / total
    print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
          f"Loss: {total_loss/len(train_loader):.4f} | "
          f"Train Acc: {train_acc:.2f}% | "
          f"LR: {scheduler.get_last_lr()[0]:.6f}")

# ── Test accuracy ─────────────────────────────────────────────
model.eval()
correct = 0
total   = 0

with torch.no_grad():
    for data, targets in test_loader:
        data       = data.view(data.size(0), -1).to(device)
        spike_data = spikegen.rate(data, num_steps=NUM_STEPS)
        spike_out, mem_out = model(spike_data)
        pred     = spike_out.sum(0).argmax(dim=1)
        correct += (pred == targets).sum().item()
        total   += targets.size(0)

test_acc = 100 * correct / total
print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
print(f"Paper reports 79-85% for LIF model")

# ── Verilog parameters from fixed values ──────────────────────
print(f"\n{'=' * 60}")
print("Verilog parameters — fixed values")
print(f"{'=' * 60}")
print(f"  Beta = {BETA}")
print(f"    → closest shift: membrane - (membrane >>> 4) = 0.9375")
print(f"    → error vs trained: {abs(BETA - 0.9375):.4f}")
print(f"  Threshold = {THRESHOLD}")
print(f"    → hardware Q1.15 integer: {int(THRESHOLD * WEIGHT_SCALE)}")
print(f"\n  Update lif.sv:")
print(f"    assign after_leak = membrane - (membrane >>> 4);")
print(f"    parameter THRESHOLD = {int(THRESHOLD * WEIGHT_SCALE)};")

# ── Export weights ────────────────────────────────────────────
os.makedirs('mem_files', exist_ok=True)

def export_q1_15(weight_tensor, filename):
    weights     = weight_tensor.detach().cpu().numpy()
    weights_q15 = np.clip(
        np.round(weights * WEIGHT_SCALE),
        -32768, 32767).astype(np.int32)
    with open(filename, 'w') as f:
        for row in weights_q15:
            for val in row:
                f.write(f"{int(val) & 0xFFFF:04X}\n")
    print(f"  Saved: {filename}  shape={weights_q15.shape}  "
          f"min={weights_q15.min()}  max={weights_q15.max()}")
    return weights_q15

print(f"\n{'=' * 60}")
print("Exporting Q1.15 weights — paper Section 4.3")
print(f"{'=' * 60}")
w1 = export_q1_15(model.fc1.weight, 'mem_files/weights_layer1.mem')
w2 = export_q1_15(model.fc2.weight, 'mem_files/weights_layer2.mem')

# ── Test sample trace ─────────────────────────────────────────
print(f"\n{'=' * 60}")
print("Test sample trace for Verilog testbench")
print(f"{'=' * 60}")

model.eval()
with torch.no_grad():
    sample_data, sample_label = test_dataset[0]
    sample_flat  = sample_data.view(1, -1)
    spike_train  = spikegen.rate(sample_flat, num_steps=NUM_STEPS)
    spike_out, mem_out = model(spike_train)
    pred = spike_out.sum(0).argmax(dim=1).item()

    t0_spikes = spike_train[0, 0, :].numpy().astype(int)
    print(f"  True label       : {sample_label}")
    print(f"  Software predicts: {pred}")
    print(f"  Correct          : {'YES ✅' if pred==sample_label else 'NO ❌'}")
    print(f"  Spike density t0 : {t0_spikes.mean():.3f} "
          f"({t0_spikes.sum()} of 784 active)")

    with open('mem_files/test_spike_t0.mem', 'w') as f:
        for chunk in range(0, INPUT_SIZE, 16):
            word = 0
            for b in range(16):
                if chunk + b < INPUT_SIZE:
                    word |= (int(t0_spikes[chunk+b]) << b)
            f.write(f"{word:04X}\n")
    print(f"  Saved: mem_files/test_spike_t0.mem")

print(f"\n{'=' * 60}")
print("All files saved — copy mem_files/ to Vivado project")
print(f"{'=' * 60}")
print(f"  weights_layer1.mem  {HIDDEN_SIZE}×{INPUT_SIZE} = "
      f"{HIDDEN_SIZE*INPUT_SIZE} entries")
print(f"  weights_layer2.mem  {OUTPUT_SIZE}×{HIDDEN_SIZE} = "
      f"{OUTPUT_SIZE*HIDDEN_SIZE} entries")
print(f"  test_spike_t0.mem   timestep 0 spike pattern")
