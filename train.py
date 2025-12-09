import os
import time
from typing import Dict, Tuple, List
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm
from models.cic_2018_model import Cic_2018_1D
from models.cic_2019_model import Cic_2019_1D
from models.unsw_nb15_model import unsw_nb15_1D
from tools.center_crop import device
from tools.dataloader import Cic_2018_Dataset, UNSWNB15, Cic_2019_dataset
from config import get_config
from tools.earlystop import EarlyStopping
from tools.file_name_cic2018 import parse_cic_filename
from tools.file_name_cic2019 import parse_dataset_name

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True


def train_epoch(model, loader, criterion, optimizer, num_epochs, epoch):
    model.train()
    for xb, yb in tqdm(loader, desc=f"Training:{epoch}|{num_epochs}", leave=False):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()


def evaluate(model, loader, cfg, str_split, epoch):
    model.eval()
    start = time.time()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for xb, yb in tqdm(loader, desc=f"{str_split}"):
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            all_preds.append(preds.argmax(dim=1).cpu())
            all_labels.append(yb.cpu())

    elapsed = time.time() - start
    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }

    os.makedirs(cfg.log_dir, exist_ok=True)
    if cfg.data_name[:7] == "cic2018":
        log_name, _ = parse_cic_filename(cfg.data_name)
        logpath_middle = os.path.join(cfg.log_dir, log_name, cfg.data_name)
    if cfg.data_name[:7] == "cic2019":
        log_name, _ = parse_dataset_name(cfg.data_name)
        logpath_middle = os.path.join(cfg.log_dir, log_name, cfg.data_name)
    if cfg.data_name == "unsw_nb15":
        logpath_middle = os.path.join(cfg.log_dir, cfg.data_name)
    os.makedirs(logpath_middle, exist_ok=True)
    log_path = os.path.join(logpath_middle, f"{cfg.data_name}.txt")

    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"===============================================================\n")
        f.write(f"---------------------->🚀[{str_split}-epoch：{epoch+1}]🚀<------------------------\n")
        f.write(f"Evaluation Time: {elapsed:.2f} seconds\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1']:.4f}\n\n")

    return metrics, elapsed


def evaluate_loss(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            total_loss += loss.item() * xb.size(0)
            count += xb.size(0)
    return total_loss / count


def run_kfold(x_all, cfg):
    kfold = KFold(n_splits=cfg.num_folds, shuffle=True, random_state=cfg.random_seed)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(x_all)):
        print(f"\n===== Fold {fold + 1} =====")
        # Prepare datasets
        train_ds = Cic_2018_Dataset(data_dir=cfg.data_dir, data_name=cfg.data_name, split='train',
                                    fold_indices=(train_idx, val_idx))
        val_ds = Cic_2018_Dataset(data_dir=cfg.data_dir, data_name=cfg.data_name, split='val',
                                  fold_indices=(train_idx, val_idx))
        test_ds = Cic_2018_Dataset(data_dir=cfg.data_dir, data_name=cfg.data_name, split='test')
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)
        test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)

        # Model, optimizer, criterion
        model = Cic_2018_1D(input_channels=cfg.input_channels, num_classes=cfg.num_classes or train_ds.num_classes).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr) if cfg.optimizer.lower() == 'adamw' else optim.Adam(model.parameters(), lr=cfg.lr)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(cfg.num_epochs):
            t0 = time.time()
            train_epoch(model, train_loader, criterion, optimizer, cfg.num_epochs, epoch)
            evaluate(model, val_loader, cfg, "val", epoch)
            metrics, pred_time = evaluate(model, test_loader, cfg, "test", epoch)
            train_time = time.time() - t0


        # Early stopping
        # val_loss = evaluate_loss(model, val_loader, criterion)
        # log_name, _ = parse_cic_filename(cfg.data_name)
        # logpath_middle = os.path.join(cfg.log_dir, log_name, cfg.data_name)
        # os.makedirs(logpath_middle, exist_ok=True)
        # ckpt_path = os.path.join(logpath_middle, f"model_fold{fold + 1}.pt")
        # early_stopper = EarlyStopping(patience=cfg.patience, verbose=True, save_path=ckpt_path)
        # print(f"Fold {fold + 1} Epoch {epoch + 1}/{cfg.num_epochs} Validation Loss: {val_loss:.4f}")
        # early_stopper(val_loss, model)
        # if early_stopper.early_stop:
        #     print("Early stopping triggered.\n")
        #     break

        break  # 仅做一次


def run_experiment(cfg,
                   train_loader,
                   val_loader,
                   test_loader,
                   device) -> Tuple[List[Dict], Dict]:
    if cfg.data_name[:7] == "cic2019":
        model = Cic_2019_1D(input_channels=cfg.input_channels, num_classes=cfg.num_classes or train_loader.dataset.num_classes).to(device)
    else:
        model = unsw_nb15_1D(input_channels=cfg.input_channels,
                             num_classes=cfg.num_classes or train_loader.dataset.num_classes).to(device)

    optimizer = (optim.AdamW(model.parameters(), lr=cfg.lr)
                 if cfg.optimizer.lower() == "adamw"
                 else optim.Adam(model.parameters(), lr=cfg.lr))

    criterion = nn.CrossEntropyLoss()
    history: list[dict] = []
    best_val_score = -float("inf")
    best_metrics = None

    for epoch in range(cfg.num_epochs):
        t0 = time.time()

        train_epoch(model, train_loader, criterion, optimizer,
                    cfg.num_epochs, epoch)

        val_metrics, _ = evaluate(model, val_loader, cfg, "val", epoch)
        test_metrics, pred_time = evaluate(model, test_loader, cfg,"test", epoch)

        elapsed = time.time() - t0

        # 收集日志
        record = {
            "epoch": epoch + 1,
            "val":   val_metrics,
            "test":  test_metrics,
            "time":  round(elapsed, 2)
        }
        history.append(record)

        val_key = next(iter(val_metrics))
        if val_metrics[val_key] > best_val_score:
            best_val_score = val_metrics[val_key]
            best_metrics = test_metrics.copy()

        print(f"[Epoch {epoch+1}/{cfg.num_epochs}] "
              f"val_{val_key}: {val_metrics[val_key]:.4f} | "
              f"time: {elapsed:.2f}s")

    print("Training completed ✅")
    return history, best_metrics or {}


def train():
    cfg = get_config()
    if cfg.data_name[:7] == "cic2018":
        base_ds = Cic_2018_Dataset(data_dir=cfg.data_dir, data_name=cfg.data_name, split='train')
        x_all = base_ds.x_data.cpu().numpy()
        run_kfold(x_all, cfg)
    if cfg.data_name[:7] == "cic2019":
        train_ds = Cic_2019_dataset(data_dir=cfg.data_dir, data_name=cfg.data_name, split="train", device="cuda")
        val_ds = Cic_2019_dataset(data_dir=cfg.data_dir, data_name=cfg.data_name, split="val", device="cuda")
        test_ds = Cic_2019_dataset(data_dir=cfg.data_dir, data_name=cfg.data_name, split="test", device="cuda")
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)
        run_experiment(cfg, train_loader, val_loader, test_loader, device="cuda")
    else:
        assert cfg.data_name == "unsw_nb15", "data name Error!"
        train_ds = UNSWNB15(split="train", device="cuda")
        val_ds = UNSWNB15(split="val", device="cuda")
        test_ds = UNSWNB15(split="test", device="cuda")
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)
        run_experiment(cfg, train_loader, val_loader, test_loader, device="cuda")


if __name__ == "__main__":
    train()
