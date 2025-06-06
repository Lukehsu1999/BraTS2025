import os
import wandb
import numpy as np
import torch
import torch.nn as nn
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.transforms import Activations, AsDiscrete
from monai.metrics import DiceMetric
from functools import partial
from monai_brats_dataloader import build_loader_from_csv
from swin_unetr import build_swin_unetr
from tqdm import tqdm, trange
from monai.data import decollate_batch
# from train_utils import (
#     train_epoch, val_epoch, trainer, save_checkpoint
# )

# ==== Config ====
class Config:
    exp_name = "T7-swin-unetr-exp1"
    
    # wandb settings
    wandb_project = "BraTS"
    wandb_run_name = exp_name
    wandb_enabled = True  # set to False if you want to disable logging temporarily

    meta_csv = "./meta/data_split_Usage20.csv"
    data_dir = "/media/volume1/BraTS2025/7/MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth/MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth"
    save_dir = "./experiments/"+exp_name
    
    # Model parameters
    roi = (128, 128, 128)
    in_channels = 4
    out_channels = 4
    feature_size = 48
    
    # Training
    batch_size = 2
    num_workers = 4
    pin_memory = True
    max_epochs = 300
    lr = 1e-4
    weight_decay = 1e-5
    val_every = 1
    sw_batch_size = 1
    infer_overlap = 0.6
    
    # wandb settings
    wandb_project = "BraTS"
    wandb_run_name = "T7-swin-unetr-exp1"
    wandb_enabled = True  # set to False if you want to disable logging temporarily

# ====== Helper ======
def save_checkpoint(model, epoch, best_dice, save_path):
    state_dict = model.state_dict()
    torch.save({"epoch": epoch, "best_dice": best_dice, "state_dict": state_dict}, save_path)
    print("💾 Saved best model to:", save_path)

# ====== Training and Validation ======
def train_epoch(model, loader, optimizer, loss_func, device):
    model.train()
    epoch_loss = 0
    for batch in tqdm(loader, desc="Train"):
        inputs = batch["image"].to(device)
        targets = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss / len(loader)

def val_epoch(model, loader, inferer, metric, post_sigmoid, post_pred, device):
    model.eval()
    metric.reset()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Val"):
            inputs = batch["image"].to(device)
            targets = batch["label"].to(device)

            outputs = inferer(inputs)
            outputs = [post_pred(post_sigmoid(o)) for o in decollate_batch(outputs)]
            targets = decollate_batch(targets)
            metric(y_pred=outputs, y=targets)

    dsc, not_nans = metric.aggregate()
    return dsc.cpu().numpy()

def main():
    config = Config()
    
    # save dir
    os.makedirs(config.save_dir, exist_ok=True)

    # ==== Initialize wandb ====
    if config.wandb_enabled:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config={
                "roi": config.roi,
                "batch_size": config.batch_size,
                "in_channels": config.in_channels,
                "out_channels": config.out_channels,
                "lr": config.lr,
                "weight_decay": config.weight_decay,
                "sw_batch_size": config.sw_batch_size,
                "infer_overlap": config.infer_overlap,
                "max_epochs": config.max_epochs,
            },
            reinit=True,
            settings=wandb.Settings(code_dir=""),
        )

    # ==== Build loaders ====
    print("[🚀] Loading data...")
    train_loader, val_loader, test_loader = build_loader_from_csv(
        csv_path=config.meta_csv,
        data_dir=config.data_dir,
        roi=config.roi,
        out_channels=config.out_channels,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    # ==== Log counts ====
    train_cnt = len(train_loader.dataset)
    val_cnt = len(val_loader.dataset)
    test_cnt = len(test_loader.dataset)

    print(f"[📊] Train: {train_cnt} | Val: {val_cnt} | Test: {test_cnt}")
    if config.wandb_enabled:
        wandb.config.update({
            "train_samples": train_cnt,
            "val_samples": val_cnt,
            "test_samples": test_cnt,
        })

    # ==== Sample check ====
    print("[🔍] Fetching one sample from train loader...")
    sample_batch = next(iter(train_loader))
    print(f"Image shape: {sample_batch['image'].shape}")
    print(f"Label shape: {sample_batch['label'].shape}")
    
    # set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    
    # load model
    model = build_swin_unetr(
        img_size=config.roi,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        feature_size=config.feature_size,
    ).to(device)
    
    # ⚙️ Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epochs)
    
    # 🎯 Loss & Metrics
    loss_func = DiceLoss(to_onehot_y=False, softmax=True)
    
    # Post-processing
    post_sigmoid = Activations(softmax=True)
    post_pred = AsDiscrete(argmax=True, to_onehot=config.out_channels)
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=True)
    
    # Inference
    model_inferer = partial(
        sliding_window_inference,
        roi_size=config.roi,
        sw_batch_size=config.sw_batch_size,
        predictor=model,
        overlap=config.infer_overlap,
    )
    
    best_dice = 0.0
    for epoch in trange(config.max_epochs, desc="Epochs"):
        print(f"\n🌀 Epoch {epoch}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, loss_func, device)
        print(f"✅ Train Loss: {train_loss:.4f}")

        # Val
        if (epoch + 1) % config.val_every == 0 or epoch == 0:
            val_dice = val_epoch(model, val_loader, model_inferer, dice_metric, post_sigmoid, post_pred, device)
            avg_dice = np.mean(val_dice[1:])  # exclude BG
            print(f"🎯 Val Dice: {avg_dice:.4f}")
            for i, name in enumerate(["BG", "NETC", "SNFH", "ET"]):
                print(f"   {name:<5}: {val_dice[i]:.4f}")

            # Save best
            if avg_dice > best_dice:
                best_dice = avg_dice
                save_checkpoint(model, epoch, best_dice, os.path.join(config.save_dir, "best_model.pt"))
        
        if config.wandb_enabled:
            wandb.log({
                "train/train_loss": train_loss,
                "val/val_dice_avg": avg_dice,
                "val/val_dice_NETC": val_dice[1],
                "val/val_dice_SNFH": val_dice[2],
                "val/val_dice_ET": val_dice[3],
                "lr": scheduler.get_last_lr()[0]
            }, step=epoch)

        scheduler.step()



if __name__ == "__main__":
    main()