import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os

# ---------------- CONFIG ----------------
CSV_PATH = "dataset/az_handwritten.csv"
IMG_SIZE = 28
BATCH_SIZE = 64        
EPOCHS = 10
DEVICE = "cpu"

# ---------------- LOAD CSV ----------------
print("Loading CSV...")
data = pd.read_csv(CSV_PATH, header=None)

labels = data.iloc[:, 0].values
images = data.iloc[:, 1:].values

# Normalize and reshape
images = images / 255.0
images = images.reshape(-1, 1, IMG_SIZE, IMG_SIZE)

# Convert to tensors
X = torch.tensor(images, dtype=torch.float32)   #img tensor
y = torch.tensor(labels, dtype=torch.long)  #label tensor

# Dataset & DataLoader 
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ---------------- MODEL ----------------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3),     #26x26
            nn.ReLU(),         
            nn.MaxPool2d(2),    #13x13

            nn.Conv2d(32, 64, 3),    #11x11
            nn.ReLU(),
            nn.MaxPool2d(2),    #5x5

            nn.Conv2d(64, 128, 3),   #3x3
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 3 * 3, 128),     #1152 extracted features are compressed into 128 learned feature representations.
            nn.ReLU(),
            nn.Linear(128, 26)     
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  #flattening -transforming the feature map into a single column
        return self.fc(x)

model = CNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---------------- TRAIN ----------------
print("Training started...")
for epoch in range(EPOCHS):
    total_loss = 0
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(batch_x)  #forward pass the model makes guess
        loss = criterion(outputs, batch_y)  
        loss.backward()  #backpropagation
        optimizer.step()   #optimization- weights are adjusted

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss:.4f}")

# ---------------- SAVE ----------------
os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), "model/handwriting_model.pth")
print("Model saved as model/handwriting_model.pth")
