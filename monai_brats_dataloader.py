import os
import pandas as pd
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ConcatItemsd,
    CropForegroundd, RandSpatialCropd, RandFlipd,
    NormalizeIntensityd, RandScaleIntensityd, RandShiftIntensityd,
    AsDiscreted
)

def build_loader_from_csv(
    csv_path,
    data_dir,
    roi,
    out_channels,
    batch_size=2,
    num_workers=4,
    pin_memory=True,
):
    df = pd.read_csv(csv_path)
    train_df = df[df["data_split"] == "Train"]
    val_df = df[df["data_split"] == "Validation"]
    test_df = df[df["data_split"] == "Test"]

    def make_file_list(subj_df):
        subjects = []
        for subj in subj_df["folder_name"]:
            subj_dir = os.path.join(data_dir, subj)
            sample = {
                "image_t1": os.path.join(subj_dir, f"{subj}-t1n.nii.gz"),
                "image_t1ce": os.path.join(subj_dir, f"{subj}-t1c.nii.gz"),
                "image_t2": os.path.join(subj_dir, f"{subj}-t2w.nii.gz"),
                "image_t2f": os.path.join(subj_dir, f"{subj}-t2f.nii.gz"),
                "label": os.path.join(subj_dir, f"{subj}-seg.nii.gz"),
            }
            if all(os.path.exists(p) for p in sample.values()):
                subjects.append(sample)
        return subjects

    train_files = make_file_list(train_df)
    val_files = make_file_list(val_df)
    test_files = make_file_list(test_df)

    keys = ["image_t1", "image_t1ce", "image_t2", "image_t2f"]
    common_transforms = [
        LoadImaged(keys=keys + ["label"]),
        EnsureChannelFirstd(keys=keys + ["label"]),
        ConcatItemsd(keys=keys, name="image"),
        AsDiscreted(keys="label", to_onehot=out_channels),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]

    train_transform = Compose(common_transforms + [
        CropForegroundd(keys=["image", "label"], source_key="image", k_divisible=roi, allow_smaller=True),
        RandSpatialCropd(keys=["image", "label"], roi_size=roi, random_size=False),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
    ])

    val_transform = Compose(common_transforms)

    train_ds = Dataset(data=train_files, transform=train_transform)
    val_ds = Dataset(data=val_files, transform=val_transform)
    test_ds = Dataset(data=test_files, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader