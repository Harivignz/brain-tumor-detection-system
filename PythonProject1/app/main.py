import os
import numpy as np
import torch
import torch.nn as nn
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
import random

# ========== Stage Detection CNN ==========
class TumorStageCNN(nn.Module):
    def __init__(self):
        super(TumorStageCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(2)

        dummy_input = torch.zeros(1, 1, 128, 128, 128)
        x = self.pool(torch.relu(self.conv1(dummy_input)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool2(torch.relu(self.conv3(x)))
        self.flatten_size = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool2(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ========== U-Net for Segmentation ==========
class UNet3D(nn.Module):
    def __init__(self):
        super(UNet3D, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        self.decoder = nn.Sequential(
            nn.Conv3d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.upsample = nn.Upsample(size=(128, 128, 128), mode='trilinear', align_corners=True)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.upsample(x)
        return x

# ========== Tumor Prediction (Random for now) ==========
def predict_tumor_random():
    return random.choice([True, False])

# ========== Load 3D MRI NPY ==========
def load_mri(filepath):
    image = np.load(filepath).astype(np.float32)
    image = np.expand_dims(image, axis=0)  # [C, D, H, W]
    tensor = torch.tensor(image).unsqueeze(0)  # [1, C, D, H, W]
    return tensor

# ========== GUI File Picker ==========
def select_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select a .npy MRI Scan", filetypes=[("NumPy files", "*.npy")])
    return file_path

# ========== Main Execution ==========
def main():
    file_path = select_file()
    if not file_path:
        messagebox.showwarning("No File", "❌ No file selected.")
        return

    is_tumor = predict_tumor_random()
    if not is_tumor:
        print("✅ No Tumor detected.")
        messagebox.showinfo("Result", "✅ No Tumor detected.")
        return

    # Load MRI tensor
    image_tensor = load_mri(file_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = image_tensor.to(device)

    # ========== Run Segmentation ==========
    unet_model = UNet3D().to(device)
    unet_model.load_state_dict(torch.load("brain_tumor_unet_3d.pth", map_location=device))
    unet_model.eval()
    with torch.no_grad():
        mask = unet_model(image_tensor).cpu().numpy().squeeze()

    # ========== Show Segmentation ==========
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image_tensor.cpu().numpy()[0, 0, :, :, 64], cmap='gray')
    plt.title("MRI Slice (Middle)")

    plt.subplot(1, 2, 2)
    plt.imshow(mask[:, :, 64], cmap='hot')
    plt.title("Predicted Tumor Mask")

    plt.show()

    # ========== Run Stage Classification ==========
    stage_model = TumorStageCNN().to(device)
    stage_model.load_state_dict(torch.load("tumor_stage_cnn.pth", map_location=device))
    stage_model.eval()
    with torch.no_grad():
        output = stage_model(image_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        stage = torch.argmax(probs).item()

    label_map = {0: "Grade I", 1: "Grade II", 2: "Grade III", 3: "Grade IV"}
    result = f"🎯 Predicted Tumor Stage: {label_map[stage]}"
    print(result)
    messagebox.showinfo("Tumor Stage", result)

# Run It
if __name__ == "__main__":
    main()
