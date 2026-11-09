import argparse
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import KinshipDataset, split_families
from model import TriSubjectKinshipNet


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for f, m, c, y in tqdm(loader, desc="Training", leave=False):
        f, m, c, y = f.to(device), m.to(device), c.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(f, m, c)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend(probs)
        all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader)
    all_preds = np.array(all_preds).reshape(-1)
    all_labels = np.array(all_labels).reshape(-1)
    acc = accuracy_score(all_labels, (all_preds >= 0.5).astype(int))
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for f, m, c, y in tqdm(loader, desc="Validating", leave=False):
        f, m, c, y = f.to(device), m.to(device), c.to(device), y.to(device)

        logits = model(f, m, c)
        loss = criterion(logits, y)

        total_loss += loss.item()
        probs = torch.sigmoid(logits).cpu().numpy()
        all_preds.extend(probs)
        all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader)
    all_preds = np.array(all_preds).reshape(-1)
    all_labels = np.array(all_labels).reshape(-1)
    acc = accuracy_score(all_labels, (all_preds >= 0.5).astype(int))

    try:
        auc = roc_auc_score(all_labels, all_preds)
        if math.isnan(auc):
            auc = 0.0
    except Exception:
        auc = 0.0

    return avg_loss, acc, auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs_stage1", type=int, default=10)
    parser.add_argument("--epochs_stage2", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--save_dir", type=Path, default=Path("weights"))
    args = parser.parse_args()

    args.save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"[+] Running training on: {device}")

    train_df, val_df, _ = split_families(
        Path("data/triplets.csv"), Path("data/aligned")
    )
    train_ds = KinshipDataset(train_df, is_train=True)
    val_ds = KinshipDataset(val_df, is_train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    model = TriSubjectKinshipNet(freeze_backbone=True).to(device)
    criterion = nn.BCEWithLogitsLoss()

    best_score = -1.0

    print("\n--- STAGE 1: Training Classifier Head (Backbone Frozen) ---")
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(), lr=1e-3, weight_decay=1e-4
    )

    for epoch in range(1, args.epochs_stage1 + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch:02d} | Train Loss: {tr_loss:.4f} Acc: {tr_acc:.3f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} AUC: {val_auc:.3f}"
        )

        score = val_auc if val_auc > 0.0 else val_acc
        torch.save(model.state_dict(), args.save_dir / "latest_model.pt")

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), args.save_dir / "best_model.pt")
            print(
                f"  [+] Saved new checkpoint: best_model.pt (score: {best_score:.4f})"
            )

    print("\n--- STAGE 2: End-to-End Fine-Tuning ---")
    for param in model.backbone.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": 1e-5},
            {"params": model.classifier.parameters(), "lr": 1e-4},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs_stage2
    )

    for epoch in range(1, args.epochs_stage2 + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        print(
            f"Epoch {epoch:02d} | Train Loss: {tr_loss:.4f} Acc: {tr_acc:.3f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} AUC: {val_auc:.3f}"
        )

        score = val_auc if val_auc > 0.0 else val_acc
        torch.save(model.state_dict(), args.save_dir / "latest_model.pt")

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), args.save_dir / "best_model.pt")
            print(
                f"  [+] Saved new checkpoint: best_model.pt (score: {best_score:.4f})"
            )

    print(f"\n[+] Training complete. Best validation score: {best_score:.4f}")
    print(f"[+] Checkpoints saved in: {args.save_dir.resolve()}")


if __name__ == "__main__":
    main()
