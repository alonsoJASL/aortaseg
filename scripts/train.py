# src/train.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from aortaseg.aorta_unet import UNet
from aortaseg.dataset import get_dataloader

import argparse 

def main(args) : 
    # Set your data paths
    train_image_folder = args.image_folder
    train_mask_folder = args.mask_folder

    # Set hyperparameters
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    epochs = args.epochs

    # Create U-Net model and move it to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, out_channels=1).to(device)  # Assuming grayscale images
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    # Get dataloader with torchio
    image_paths = [os.path.join(train_image_folder, file) for file in os.listdir(train_image_folder)]
    mask_paths = [os.path.join(train_mask_folder, file) for file in os.listdir(train_mask_folder)]

    dataloader = get_dataloader(image_paths, mask_paths, batch_size)

    # Training loop
    for epoch in range(epochs):
        for batch in dataloader:
            inputs = batch["image"]["data"].to(device)
            targets = batch["mask"]["data"].to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    # Save the trained model
    torch.save(model.state_dict(), args.model_path)

if __name__ == "__main__" :
    input_parser = argparse.ArgumentParser(description="Train aorta segmentation model")
    input_parser.add_argument("--image-folder", type=str, help="Path to training images")
    input_parser.add_argument("--mask-folder", type=str, default="aorta_model.pth", help="Path to training masks")
    input_parser.add_argument("--model-path", type=str, help="Path to save model")
    input_parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    input_parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    input_parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")

    args = input_parser.parse_args()
    main(args)