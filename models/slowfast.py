import os
import torch
from torch import nn
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MeanMetric, SymmetricMeanAbsolutePercentageError
from utils.checkpoint import Checkpoint
from configs.config import args

checkpoint = Checkpoint()

# Load pretrained model and replace final head for regression
slow_fast_model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
in_features = slow_fast_model.blocks[-1].proj.in_features
slow_fast_model.blocks[-1].proj = nn.Linear(in_features, 1)

def pad_to_32(data: torch.Tensor):
    T = data.shape[2]
    if T == 32:
        return data
    elif T < 32:
        padd = 32 - T
        repeat_frame = data[:, :, -1:, :, :].repeat(1, 1, padd, 1, 1)
        return torch.cat([repeat_frame, data], dim=2)
    else:
        return data[:, :, :32, :, :]

def pack_pathway_output(frames, alpha):
    fast_pathway = frames
    slow_pathway = frames[:, :, ::alpha, :, :]
    return [slow_pathway, fast_pathway]

def train_and_evaluate(model, model_name: str, train_loader, val_loader,
                      loss_fn, num_epochs: int, alpha: int = 4):

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    model.to(args.device)

    train_losses, val_losses = [], []
    train_smapes, val_smapes = [], []

    best_loss = float("inf")
    patience = 5
    counter = 0

    for epoch in range(num_epochs):
        # ========== TRAINING ==========
        model.train()
        train_loss = MeanMetric().to(args.device)
        train_smape = SymmetricMeanAbsolutePercentageError().to(args.device)

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            x, y = x.to(args.device), y.to(args.device)
            x = pad_to_32(x)
            x = pack_pathway_output(x, alpha)

            optimizer.zero_grad()
            y_pred = model(x).squeeze()
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()

            train_loss.update(loss)
            train_smape.update(y_pred, y)  

        avg_train_loss = train_loss.compute().item()
        avg_train_smape = train_smape.compute().item()
        train_losses.append(avg_train_loss)
        train_smapes.append(avg_train_smape)

        # ========== VALIDATION ==========
        model.eval()
        val_loss = MeanMetric().to(args.device)
        val_smape = SymmetricMeanAbsolutePercentageError().to(args.device)

        with torch.inference_mode():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                x , y = x.to(args.device), y.to(args.device)
                x = pad_to_32(x)
                x = pack_pathway_output(x, alpha)
                y_pred = model(x).squeeze()
                loss = loss_fn(y_pred, y)

                val_loss.update(loss)
                val_smape.update(y_pred, y)  # ✅ CORRECT

        avg_val_loss = val_loss.compute().item()
        avg_val_smape = val_smape.compute().item()
        val_losses.append(avg_val_loss)
        val_smapes.append(avg_val_smape)

        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}, SMAPE: {avg_train_smape:.2f} | "
              f"Val Loss: {avg_val_loss:.4f}, SMAPE: {avg_val_smape:.2f}")

        if avg_val_loss < best_loss:
            checkpoint.save(model, optimizer, train_losses, val_losses, avg_val_smape, model_name, epoch)
            print("Model saved.")
            best_loss = avg_val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("No improvement in validation loss. Early stopping triggered.")
                break

        scheduler.step(avg_val_loss)

    return train_losses, train_smapes, val_losses, val_smapes
