"""
Simple TCN (Temporal Convolutional Network) for Phase Segmentation

Lightweight causal TCN: 4 layers, ~15K parameters
Input: [B, 6, T] IMU sequence within active segment
Output: [B, 2, T] (concentric/eccentric probabilities)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Sequence, Tuple


class CausalConv1d(nn.Module):
    """Causal 1D convolution with appropriate padding."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
    
    def forward(self, x):
        # x: [B, C, T]
        x = F.pad(x, (self.padding, 0))  # causal: pad left only
        return self.conv(x)


class ResidualBlock(nn.Module):
    """TCN residual block with causal conv + dropout."""
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: [B, C, T]
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        # Residual connection (same length)
        if out.shape[-1] == x.shape[-1]:
            return out + x
        return out


class SimpleTCN(nn.Module):
    """4-layer causal TCN for phase segmentation."""
    def __init__(
        self,
        input_channels: int = 6,
        num_classes: int = 2,
        num_filters: int = 32,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        # Input projection: map input_channels -> num_filters
        self.input_conv = nn.Conv1d(input_channels, num_filters, 1)
        
        # TCN layers (all num_filters -> num_filters)
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            layers.append(ResidualBlock(num_filters, kernel_size, dilation, dropout))
        
        self.tcn_layers = nn.ModuleList(layers)
        self.conv1x1 = nn.Conv1d(num_filters, num_classes, 1)
    
    def forward(self, x):
        # x: [B, C, T]
        out = self.input_conv(x)  # [B, num_filters, T]
        for layer in self.tcn_layers:
            out = layer(out)
        logits = self.conv1x1(out)  # [B, num_classes, T]
        return logits


class PhaseSegmentDataset(Dataset):
    """Dataset for TCN phase segmentation."""
    def __init__(self, sequences, labels, slice_len=300):
        """
        sequences: List of [T, C] numpy arrays (active segments)
        labels: List of [T] numpy arrays (0=eccentric, 1=concentric, -1=pad)
        slice_len: Fixed length for each sample
        """
        self.samples = []
        for seq, lab in zip(sequences, labels):
            # Extract overlapping slices
            n = len(seq)
            if n <= slice_len:
                # Pad
                padded_seq = np.pad(seq, ((0, max(0, slice_len - n)), (0, 0)), mode='edge')
                padded_lab = np.pad(lab, (0, max(0, slice_len - n)), constant_values=-1)
                self.samples.append((padded_seq[:slice_len], padded_lab[:slice_len]))
            else:
                # Sliding window
                stride = slice_len // 2
                for start in range(0, n - slice_len + 1, stride):
                    self.samples.append((seq[start:start+slice_len], lab[start:start+slice_len]))
                # Last slice
                if start + slice_len < n:
                    self.samples.append((seq[-slice_len:], lab[-slice_len:]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        seq, lab = self.samples[idx]
        # [T, C] -> [C, T]
        x = torch.from_numpy(seq).float().transpose(0, 1)
        y = torch.from_numpy(lab).long()
        return x, y


def extract_active_segments_for_tcn(train_streams, imu_columns):
    """Extract active segments and phase labels for TCN training."""
    sequences = []
    labels = []
    
    for _, df in train_streams:
        if "phase" not in df.columns:
            continue
        
        x = df[list(imu_columns)].to_numpy(dtype=np.float32)
        phase_labels = df["phase"].to_numpy()
        
        # Find active segments (concentric/eccentric)
        active_mask = np.array([str(p) in {"concentric", "eccentric"} for p in phase_labels])
        
        # Split into contiguous active segments
        in_active = False
        seg_start = 0
        for i, is_active in enumerate(active_mask):
            if is_active and not in_active:
                seg_start = i
                in_active = True
            elif not is_active and in_active:
                # End of segment
                if i - seg_start >= 10:  # min 0.1s
                    seq = x[seg_start:i]
                    lab = np.array([1 if str(p) == "concentric" else 0 for p in phase_labels[seg_start:i]])
                    sequences.append(seq)
                    labels.append(lab)
                in_active = False
        
        # Trailing segment
        if in_active and len(active_mask) - seg_start >= 10:
            seq = x[seg_start:]
            lab = np.array([1 if str(p) == "concentric" else 0 for p in phase_labels[seg_start:]])
            sequences.append(seq)
            labels.append(lab)
    
    return sequences, labels


def train_tcn_phase_segmenter(train_streams, imu_columns, device='cpu', epochs=20, lr=1e-3, batch_size=32):
    """Train TCN phase segmenter on active segments."""
    sequences, labels = extract_active_segments_for_tcn(train_streams, imu_columns)
    
    if not sequences:
        return None
    
    print(f"      [TCN] Training on {len(sequences)} active segments")
    
    # Create dataset
    dataset = PhaseSegmentDataset(sequences, labels, slice_len=300)
    if len(dataset) == 0:
        return None
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Model
    model = SimpleTCN(input_channels=len(imu_columns), num_classes=2)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    # Training
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(x_batch)  # [B, 2, T]
            
            # Reshape for cross entropy: [B*T, 2] vs [B*T]
            B, C, T = logits.shape
            logits_flat = logits.permute(0, 2, 1).reshape(B * T, C)
            labels_flat = y_batch.reshape(B * T)
            
            # Filter out padding (-1)
            valid = labels_flat >= 0
            if valid.sum() == 0:
                continue
            
            loss = criterion(logits_flat[valid], labels_flat[valid])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"      [TCN] Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")
    
    return model


def predict_tcn_phase(model, df, active_segments, imu_columns, device='cpu'):
    """Predict phase probabilities using TCN within active segments."""
    x = df[list(imu_columns)].to_numpy(dtype=np.float32)
    n = len(x)
    
    # Default: uncertain
    phase_probs = np.ones((n, 2)) * 0.5
    
    model.eval()
    with torch.no_grad():
        for seg_start, seg_end in active_segments:
            if seg_start >= seg_end:
                continue
            seg_x = x[seg_start:seg_end]
            
            # Pad/truncate to 300 samples
            seg_len = len(seg_x)
            if seg_len < 300:
                padded = np.pad(seg_x, ((0, 300 - seg_len), (0, 0)), mode='edge')
            else:
                padded = seg_x[:300]
            
            # [T, C] -> [1, C, T]
            x_tensor = torch.from_numpy(padded).float().transpose(0, 1).unsqueeze(0).to(device)
            logits = model(x_tensor)  # [1, 2, T]
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]  # [2, T]
            
            # Take only valid portion
            valid_len = min(seg_len, 300)
            probs_valid = probs[:, :valid_len].T  # [valid_len, 2]
            
            # Map back to original indices
            end_idx = min(seg_start + valid_len, seg_end)
            phase_probs[seg_start:end_idx] = probs_valid[:end_idx - seg_start]
    
    return phase_probs
