import time
import torch
import numpy as np
import torchio as tio
from torch.utils.data import DataLoader
from tqdm import tqdm
import monai
from temporalSlice.config import TRAIN_BATCH_SIZE, VAL_BATCH_SIZE, NUM_EPOCHS, OUT_DIR, SAVE_INTERVAL
from temporalSlice.datasets import get_datasets
from temporalSlice.model import temporalSliceNet
from temporalSlice.utils import get_device

def prepare_batch(batch, device):
    inputs = torch.tensor(batch['t1']['data']).to(device)
    targets = torch.tensor(batch['mask']['data']).to(device)
    return inputs.double(), targets.double()

def train(training_dir, validation_dir):
    train_losses, val_losses = [], []

    device = get_device()
    model, optimizer, train_losses = temporalSliceNet()
    loss_fn = monai.losses.DiceLoss(to_onehot_y=True, softmax=False, sigmoid=True)
    train_dataset, val_dataset = get_datasets(training_dir, validation_dir)
    train_loader = tio.SubjectsLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = tio.SubjectsLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"Epoch {epoch}/{NUM_EPOCHS}")

        # Training Phase
        model.train()
        epoch_train_losses = []
        for batch in tqdm(train_loader):
            inputs, targets = prepare_batch(batch, device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
            epoch_train_losses.append(loss.item())

        train_losses.append(np.mean(epoch_train_losses))
        print(f"Training Loss: {train_losses[-1]:.4f}")

        # Validation Phase
        model.eval()
        epoch_val_losses = []
        with torch.no_grad():
            for batch in tqdm(val_loader):
                inputs, targets = prepare_batch(batch, device)
                logits = model(inputs)
                loss = loss_fn(logits, targets)
                epoch_val_losses.append(loss.item())

        val_losses.append(np.mean(epoch_val_losses))
        print(f"Validation Loss: {val_losses[-1]:.4f}")

        # Save model checkpoint every `SAVE_INTERVAL` epochs
        if epoch % SAVE_INTERVAL == 0:
            torch.save(model.state_dict(), OUT_DIR / f"model_epoch_{epoch}.pt")

    # Save final model
    torch.save(model.state_dict(), OUT_DIR / "model_final.pt")
