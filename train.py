import torch
import torch.nn as nn
import torch.optim as optim
from models.ppg2bpnet import PPG2BPNet
from utils import load_sample_data
import numpy as np

X_target, X_calib, y, y_min, y_max = load_sample_data()
print("X_target shape:", X_target.shape)
print("X_calib shape:", X_calib.shape)
print("y shape:", y.shape)

# 2. Model, Loss, Optimizer
model = PPG2BPNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. Train
model.train()
epochs = 25
threshold = 5.0  # mmHg threshold for being considered "accurate"
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(X_target, X_calib)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    
    # Calculate "accuracy" as percentage of samples
    # for which both SBP and DBP predictions are within threshold
    with torch.no_grad():
        # compute absolute errors and check if both are within threshold
        correct = ((torch.abs(output - y)) < threshold).all(dim=1)
        accuracy = correct.float().mean().item() * 100
    
    print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f} | Accuracy: {accuracy:.2f}%")

torch.save(model.state_dict(), "model.pt")
np.savez("bp_norm_bounds.npz", y_min=y_min.numpy(), y_max=y_max.numpy())