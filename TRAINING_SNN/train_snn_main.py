import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import snntorch as snn
from snntorch import spikegen
import numpy as np

# ==========================================
# 1. CUSTOM LIF NEURON WITH REFRACTORY CONTROLLER
# ==========================================
class CustomLIFWithRefractory(nn.Module):
    """
    A custom LIF neuron layer implementing a manual refractory mechanism
    as outlined in the research paper (5-step refractory window).
    """
    def __init__(self, num_inputs, num_neurons, beta=0.9, threshold=1.0, refractory_steps=5):
        super(CustomLIFWithRefractory, self).__init__()
        self.num_neurons = num_neurons
        self.refractory_steps = refractory_steps
        
        # Base snnTorch LIF neuron layer (configured to manual reset as we handle resetting manually)
        self.lif = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism="none", learn_beta=False, learn_threshold=False)
        
        # States initialized dynamically during the forward pass execution
        self.mem = None
        self.ref_counters = None

    def reset_states(self, batch_size, device):
        """Resets membrane potentials and refractory counters at the start of a sequence."""
        self.mem = self.lif.init_leaky()
        # Ensure membrane potential has correct batch dimension
        if self.mem.ndim == 1 or self.mem.size(0) != batch_size:
            self.mem = torch.zeros(batch_size, self.num_neurons, device=device)
        # Refractory counters: Integer counters per neuron per batch element
        self.ref_counters = torch.zeros(batch_size, self.num_neurons, dtype=torch.int32, device=device)

    def forward(self, input_current):
        """
        Processes a single time-step input current through the LIF layer 
        incorporating custom refractory behavior.
        """
        # 1. Decrement all non-zero refractory counters by 1 step
        self.ref_counters = torch.clamp(self.ref_counters - 1, min=0)
        
        # 2. Update membrane potential using standard Leaky Integrate-and-Fire equation
        # Note: Membrane updates normally regardless of refractory state per paper philosophy
        self.mem, spk = self.lif(input_current, self.mem)
        
        # 3. Identify where raw internal threshold conditions are met
        raw_spike_mask = (self.mem >= self.lif.threshold)
        
        # 4. A neuron can only fire if it is NOT currently in a refractory period (counter == 0)
        available_to_spike = (self.ref_counters == 0)
        actual_spike_mask = raw_spike_mask & available_to_spike
        
        # Generate final spike tensor output (1.0 for spike, 0.0 otherwise)
        spk_out = actual_spike_mask.float()
        
        # 5. Apply Hard Reset to zero and initialize the refractory timer to 5 steps
        self.mem = torch.where(actual_spike_mask, torch.zeros_like(self.mem), self.mem)
        self.ref_counters = torch.where(actual_spike_mask, torch.tensor(self.ref_steps_val(actual_spike_mask), device=input_current.device), self.ref_counters)
        
        return spk_out, self.mem

    def ref_steps_val(self, mask):
        # Helper to return integer assignment match
        return np.int32(self.refractory_steps)


# ==========================================
# 2. SNN NETWORK ARCHITECTURE
# ==========================================
class FPGAFriendlySNN(nn.Module):
    """
    SNN Architecture matching the paper specifications adapted for MNIST:
    Input (784) -> FC1 (784 -> 512) -> Hidden LIF -> FC2 (512 -> 10) -> Output LIF
    """
    def __init__(self, beta=0.9, threshold=1.0, num_steps=25):
        super(FPGAFriendlySNN, self).__init__()
        self.num_steps = num_steps
        
        # Synaptic weight layers
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 10)
        
        # LIF Neural layers with 5-step manual refractory management
        self.hidden_lif = CustomLIFWithRefractory(num_inputs=512, num_neurons=512, beta=beta, threshold=threshold, refractory_steps=5)
        self.output_lif = CustomLIFWithRefractory(num_inputs=10, num_neurons=10, beta=beta, threshold=threshold, refractory_steps=5)

    def forward(self, rate_encoded_in):
        # rate_encoded_in dimensions: [num_steps, batch_size, 784]
        batch_size = rate_encoded_in.size(1)
        device = rate_encoded_in.device
        
        # Reset internal states for the new presentation cycle
        self.hidden_lif.reset_states(batch_size, device)
        self.output_lif.reset_states(batch_size, device)
        
        spk2_rec = []
        mem2_rec = []
        
        # Simulate over designated sequence duration
        for step in range(self.num_steps):
            # Read single binary spike input frame
            x_step = rate_encoded_in[step] 
            
            # Layer 1 Dataflow
            cur1 = self.fc1(x_step)
            spk1, _ = self.hidden_lif(cur1)
            
            # Layer 2 Dataflow
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.output_lif(cur2)
            
            # Accumulate records over time
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)
            
        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)


# ==========================================
# 3. DATA PREPARATION
# ==========================================
def get_data_loaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    return train_loader, test_loader


# ==========================================
# 4. TRAINING & EVALUATION PIPELINE
# ==========================================
def train_epoch(model, train_loader, optimizer, criterion, num_steps, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for data, targets in train_loader:
        data, targets = data.to(device), targets.to(device)
        # Flatten images to [batch_size, 784]
        data_flattened = data.view(data.size(0), -1)
        
        # Rate coding generation -> [num_steps, batch_size, 784]
        rate_encoded = spikegen.rate(data_flattened, num_steps=num_steps)
        
        optimizer.zero_grad()
        
        # Execute forward pass
        spk_rec, mem_rec = model(rate_encoded)
        
        # Compute multi-step functional loss (summed across time steps)
        loss = torch.zeros(1, device=device)
        for step in range(num_steps):
            loss += criterion(mem_rec[step], targets)
            
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Prediction calculation based on output spike accumulation
        summed_spikes = spk_rec.sum(dim=0)
        _, predicted = summed_spikes.max(1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        
    return total_loss / len(train_loader), 100 * correct / total


def evaluate_model(model, loader, criterion, num_steps, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            data_flattened = data.view(data.size(0), -1)
            rate_encoded = spikegen.rate(data_flattened, num_steps=num_steps)
            
            spk_rec, mem_rec = model(rate_encoded)
            
            loss = torch.zeros(1, device=device)
            for step in range(num_steps):
                loss += criterion(mem_rec[step], targets)
                
            total_loss += loss.item()
            
            summed_spikes = spk_rec.sum(dim=0)
            _, predicted = summed_spikes.max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
    return total_loss / len(loader), 100 * correct / total


# ==========================================
# 5. FPGA HARDWARE EXPORT FUNCTION
# ==========================================
def export_weights_for_verilog(model, export_dir="./verilog_weights", quant_scale=32768):
    """
    Exports neural network weights/biases into plain text files mapping 
    to 16-bit Fixed-Point (Q1.15 signed representation, scaled by 2^15 = 32768) 
    compatible with FPGA Verilog memory reads ($readmemh).
    """
    os.makedirs(export_dir, exist_ok=True)
    print(f"\n[Hardware Export] Formatting weights to Fixed-Point (Q1.15) inside: '{export_dir}'...")
    
    for name, param in model.named_parameters():
        if 'weight' in name or 'bias' in name:
            # Convert weight matrix to safe CPU numpy copy
            param_np = param.detach().cpu().numpy()
            
            # Scale real weights to fit Q1.15 mapping limits [-1.0, 1.0)
            scaled_param = np.clip(param_np * quant_scale, -32768, 32767).astype(np.int16)
            
            # Treat array dimensions elegantly for flat row-major sequence maps
            flat_param = scaled_param.flatten()
            
            # Generate hex format file suitable for direct Verilog system compilation
            hex_filename = os.path.join(export_dir, f"{name.replace('.', '_')}.hex")
            with open(hex_filename, 'w') as f:
                for val in flat_param:
                    # Formulate 2's complement 16-bit hex strings
                    hex_str = f"{val & 0xFFFF:04X}"
                    f.write(f"{hex_str}\n")
                    
            print(f" -> Exported {name} (Shape: {param_np.shape}) to {hex_filename}")


# ==========================================
# 6. MAIN EXECUTION ROUTINE
# ==========================================
if __name__ == "__main__":
    # Hyperparameters Configuration
    BATCH_SIZE = 128
    EPOCHS = 5
    LEARNING_RATE = 5e-4
    NUM_STEPS = 25
    BETA = 0.9
    THRESHOLD = 1.0
    
    # Check Hardware Target acceleration context
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using runtime device platform: {device}")
    
    # Dataset Preparation
    print("Loading MNIST Dataset components...")
    train_loader, test_loader = get_data_loaders(batch_size=BATCH_SIZE)
    
    # Model instantiation
    print("Initializing Network Module with 5-step custom refractory windows...")
    model = FPGAFriendlySNN(beta=BETA, threshold=THRESHOLD, num_steps=NUM_STEPS).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # Optimization Loop
    print("\nBeginning training routine...")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, NUM_STEPS, device)
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, NUM_STEPS, device)
        
        print(f"Epoch [{epoch}/{EPOCHS}] "
              f"| Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% "
              f"| Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
        
    # Persist native PyTorch weights archive
    torch.save(model.state_dict(), "model.pth")
    print("\nStandard PyTorch state weights successfully saved to 'model.pth'")
    
    # Convert and export parameters explicitly to text arrays for Verilog modules
    export_weights_for_verilog(model, export_dir="./verilog_weights", quant_scale=32768)
    print("Execution finalized successfully.")
