import time
import torch
import numpy as np
from tqdm import tqdm, trange
from monai.data import decollate_batch

class AverageMeter:
    """Tracks and updates running averages for metrics (e.g., loss, dice)."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0
        self.avg = 0
        self.val = 0

    def update(self, val, n=1):
        if isinstance(val, np.ndarray):
            val = val.astype(np.float32)
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count
        else:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

# def train_epoch(model, loader, optimizer, epoch, loss_func, device):
#     model.train()
#     start_time = time.time()
#     run_loss = AverageMeter()

#     loop = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch}", leave=False)
#     for idx, batch_data in loop:
#         data = batch_data["image"].to(device)
#         target = batch_data["label"].to(device)

#         optimizer.zero_grad()
#         logits = model(data)
#         loss = loss_func(logits, target)
#         loss.backward()
#         optimizer.step()

#         run_loss.update(loss.item(), n=data.shape[0])

#     return run_loss.avg


def val_epoch(
    model,
    loader,
    epoch,
    acc_func,  # This should now be DiceMetric
    device,
    model_inferer=None,
    post_sigmoid=None,
    post_pred=None,
    out_channels=5
):
    model.eval()
    start_time = time.time()

    acc_func.reset()  # ✅ Moved outside loop — only reset ONCE per val_epoch

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data = batch_data["image"].to(device)
            target = batch_data["label"].to(device)

            # 🔍 Inference
            logits = model_inferer(data) if model_inferer else model(data)

            # ✅ Post-processing
            val_labels_list = decollate_batch(target)
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [
                post_pred(post_sigmoid(pred_tensor)) for pred_tensor in val_outputs_list
            ]

            # ✅ Update Dice metric
            acc_func(y_pred=val_output_convert, y=val_labels_list)

            print(f"🕒 Batch {idx+1}/{len(loader)} done in {time.time() - start_time:.2f}s")
            start_time = time.time()

    # ✅ Final aggregate after the full validation epoch
    dice_per_class, valid_counts = acc_func.aggregate()

    class_names = ["BG", "NETC", "SNFH", "ET", "RC"]
    print(f"\n📊 [Epoch {epoch}] Validation Dice Scores:")
    for i in range(out_channels):
        print(f"  {class_names[i]:<5}: {dice_per_class[i]:.4f} ({int(valid_counts[i])} valid)")

    return dice_per_class  # Return per-class Dice

def save_checkpoint(model, epoch, filename="model.pt", best_dice=0.0, dir_add="./"):
    state_dict = model.state_dict()
    save_dict = {"epoch": epoch, "best_dice": best_dice, "state_dict": state_dict}
    filename = os.path.join(dir_add, filename)
    torch.save(save_dict, filename)
    print("💾 Saving checkpoint to", filename)
    
def trainer(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    scheduler,
    max_epochs,
    val_every,
    device,
    model_inferer=None,
    start_epoch=0,
    post_sigmoid=None,
    post_pred=None,
    checkpoint_dir=None,
    wandb_log=True,
):
    import wandb  # Safe even if not used (will only error if wandb_log=True and wandb not installed)

    val_acc_max = 0.0
    class_names = ["BG", "NETC", "SNFH", "ET"]
    dice_history = {name: [] for name in class_names}
    dice_avg_history = []
    loss_epochs = []
    trains_epoch = []

    for epoch in trange(start_epoch, max_epochs, desc="Epochs"):
        print(time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()

        # 🏋️ Training
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            epoch=epoch,
            loss_func=loss_func,
            device=device
        )
        print(
            "✅ Training Done {}/{}".format(epoch, max_epochs - 1),
            "Loss: {:.4f}".format(train_loss),
            "⏱️ Time: {:.2f}s".format(time.time() - epoch_time),
        )
        if wandb_log:
            wandb.log({"train/loss": train_loss}, step=epoch)

        # 📈 Validation
        if (epoch + 1) % val_every == 0 or epoch == 0:
            loss_epochs.append(train_loss)
            trains_epoch.append(int(epoch))
            epoch_time = time.time()

            val_acc = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                device=device,
                model_inferer=model_inferer,
                post_sigmoid=post_sigmoid,
                post_pred=post_pred,
            )

            # 🎯 Extract per-class Dice
            for i, name in enumerate(class_names):
                score = val_acc[i]
                dice_history[name].append(score)
                if wandb_log:
                    wandb.log({f"val/dice_{name.lower()}": score}, step=epoch)

            val_avg_acc = np.mean(val_acc[1:])  # exclude background
            dice_avg_history.append(val_avg_acc)

            print(
                "📊 Final Validation {}/{}: Avg Dice: {:.4f}, Time: {:.2f}s".format(
                    epoch, max_epochs - 1, val_avg_acc, time.time() - epoch_time
                )
            )
            if wandb_log:
                wandb.log({"val/dice_avg": val_avg_acc}, step=epoch)

            # 💾 Save best model
            if val_avg_acc > val_acc_max:
                print("🎉 New best! {:.6f} → {:.6f}".format(val_acc_max, val_avg_acc))
                val_acc_max = val_avg_acc
                if checkpoint_dir is not None:
                    save_checkpoint(model, epoch, best_acc=val_acc_max, dir_add=checkpoint_dir)

        scheduler.step()

    print("🏁 Training Finished! Best Avg Dice: {:.4f}".format(val_acc_max))

    return (
        val_acc_max,
        dice_history["NETC"],
        dice_history["SNFH"],
        dice_history["ET"],
        dice_avg_history,
        loss_epochs,
        trains_epoch,
    )