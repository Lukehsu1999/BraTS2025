from monai.networks.nets import SwinUNETR

# We are not redefining the SwinUNETR Class because we are already using monai's class
# We are making it a factory function

def build_swin_unetr(
    img_size=(128, 128, 128),
    in_channels=4,
    out_channels=5,
    feature_size=48,
    use_checkpoint=True,
):
    model = SwinUNETR(
        img_size=img_size,
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=use_checkpoint,
    )
    return model
