import os
import wandb
import torch
import torch.nn as nn
from monai_brats_dataloader import build_loader_from_csv
from swin_unetr import build_swin_unetr

# ==== Config ====
class Config:
    meta_csv = "./meta/data_split_UsageFull.csv"
    data_dir = "/media/volume1/BraTS2025/7/MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth/MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth"
    
    # Model parameters
    roi = (128, 128, 128)
    in_channels = 4
    out_channels = 5
    feature_size = 48
    
    batch_size = 2
    num_workers = 4
    pin_memory = True

def main():
    config = Config()

    # # ==== Initialize wandb ====
    # wandb.init(
    #     project="brats2025",
    #     name="sanity-check-loader",
    #     config={
    #         "roi": config.roi,
    #         "batch_size": config.batch_size,
    #         "out_channels": config.out_channels,
    #     }
    # )

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
    # wandb.config.update({
    #     "train_cnt": train_cnt,
    #     "val_cnt": val_cnt,
    #     "test_cnt": test_cnt,
    # })

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

if __name__ == "__main__":
    main()