# coding=gb2312
from torch.utils.data import Dataset, DataLoader
import warnings
import torch.optim as optim
from tqdm import tqdm, trange
import torch
import time
import os
from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------------------------------------------------
# options
import argparse
import options
opt = options.Options().init(argparse.ArgumentParser(description='ShadowRemoval')).parse_args()
# print(opt)


# ---------------------------------------------------------------------------------------------------------------------
# seed
from utils.random_seed import setup_random_seed
setup_random_seed(
    seed=opt.seed,
    deterministic=opt.deterministic,
    benchmark=opt.cudnn_benchmark
)


# ---------------------------------------------------------------------------------------------------------------------
# log
from utils.log import FormatterNoInfo, setup_logging
import logging
logger = logging.getLogger('Trainer')
timestamp = time.strftime('%Y%m%d_%H%M%S')
log_file = os.path.join(opt.log_dir, f"train_log_{timestamp}.log")
setup_logging(log_file)
logger.info("=" * 80)
logger.info(f"STARTING SHADOW REMOVAL TRAINING")
logger.info(f"Log file: {log_file}")
logger.info("=" * 80)
logger.info(f"Random seed set to: {opt.seed}")
logger.info(f"cudnn.deterministic: {opt.deterministic}")
logger.info(f"cudnn.benchmark: {opt.cudnn_benchmark}")
logger.info(f"Training options: {opt}")

# ---------------------------------------------------------------------------------------------------------------------
# train Dataset
from dataset import SRDTrainDataset, SRDTestDataset
from utils.work_init_fn import worker_init_fn
train_dataset = SRDTrainDataset(opt.train_data_dir, img_size=opt.img_size, augment=opt.augment)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.num_workers,
    pin_memory=True,
    worker_init_fn=worker_init_fn
)
test_dataset = SRDTestDataset(opt.test_data_dir, img_size=opt.img_size, augment=False)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=opt.num_workers,
    pin_memory=True
)


# ---------------------------------------------------------------------------------------------------------------------
# model settings & optimizer & loss
from model import StructFormer
from losses import CharbonnierLoss
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
device = torch.device('cuda' if torch.cuda.is_available() and opt.use_cuda else 'cpu')
logger.info(f"Using device: {device}")
model = StructFormer().to(device)
optimizer = optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
criterion_charbonnier = CharbonnierLoss().cuda()
psnr_metric = PeakSignalNoiseRatio(data_range=1.0).cuda()
num_epoch = opt.epoch
scheduler_cosine = CosineAnnealingLR(optimizer, T_max=num_epoch)
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=opt.warmup_epochs, after_scheduler=scheduler_cosine)
validation_freq = opt.validation_freq
best_psnr = -1
best_iter = 0
total_iter = 0
logger.info(f"Model initialized: {type(model).__name__}")
logger.info(f"Optimizer: {type(optimizer).__name__} (lr={opt.lr}, weight_decay={opt.weight_decay})")
logger.info(f"Loss function: {type(criterion_charbonnier).__name__}")
logger.info(f"Training for {num_epoch} epochs with validation every {validation_freq} iterations")


# ---------------------------------------------------------------------------------------------------------------------
# train
from utils.validate import validate
from utils.mixup import MixUp_AUG
from torch.cuda.amp import autocast, GradScaler

if opt.use_mixup:
    logger.info("Using MixUp data augmentation(starting from epoch {opt.mixup_start_epoch})")
    mixup_aug = MixUp_AUG()
else:
    mixup_aug = None

if opt.use_amp:
    logger.info("Using Mixed Precision Training (AMP)")
    scaler = GradScaler()
else:
    scaler = None

os.makedirs(opt.checkpoint_dir, exist_ok=True)
for epoch in range(num_epoch):
    model.train()
    epoch_loss = 0.0
    epoch_psnr = 0.0
    start_time = time.time()
    progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epoch}', total=len(train_dataloader))

    for batch_idx, (A, B, C) in enumerate(progress_bar):
        total_iter += 1
        A = A.cuda()
        B = B.cuda()
        C = C.cuda()
        if opt.use_mixup and mixup_aug is not None and epoch >= opt.mixup_start_epoch:
            B, A, C = mixup_aug.aug(B, A, C)
        optimizer.zero_grad()

        if opt.use_amp and scaler is not None:
            with autocast():
                outputs = model(A, C)
                loss = criterion_charbonnier(outputs, B)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(A, C)
            loss = criterion_charbonnier(outputs, B)
            loss.backward()
            optimizer.step()


        with torch.no_grad():
            outputs_float = outputs.float() if outputs.dtype != torch.float32 else outputs
            psnr_val = psnr_metric(outputs, B).item()
        epoch_loss += loss.item() * A.size(0)
        epoch_psnr += psnr_val * A.size(0)

        if batch_idx % 10 == 0:  # 10 epoch
            logger.debug(f"Epoch {epoch + 1} | Batch {batch_idx}/{len(train_dataloader)} | "
                         f"Loss: {loss.item():.4f} | PSNR: {psnr_val:.2f}")
        progress_bar.set_postfix(loss=loss.item(), psnr=psnr_val, iter=total_iter)

        if total_iter % validation_freq == 0:
            logger.info(f"Validating at iteration {total_iter}...")
            val_psnr = validate(model, test_dataloader, use_amp=opt.use_amp)
            logger.info(f"Validation results | PSNR={val_psnr:.2f}")
            print(f"\nValidation at Iter {total_iter}:  PSNR={val_psnr:.2f}")
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                best_iter = total_iter
                torch.save(model.state_dict(), opt.best_model_path)
                logger.info(f"saved new best model at iter {total_iter} with PSNR {val_psnr:.2f} to {opt.best_model_path}")
                print(f"Saved new best model at iter {total_iter} with PSNR {val_psnr:.2f}")


    epoch_loss = epoch_loss / len(train_dataset)
    epoch_psnr /= len(train_dataset)
    time_out = time.time() - start_time
    current_lr = optimizer.param_groups[0]['lr']
    logger.info(f"Epoch [{epoch + 1}/{num_epoch}] completed in {time_out:.2f}s | "
                f"Loss: {epoch_loss:.4f} | "
                f"PSNR: {epoch_psnr:.2f} | LR: {current_lr:.8f}")
    print(f'Epoch [{epoch + 1}/{num_epoch}] Loss: {epoch_loss:.4f}, '
          f'PSNR: {epoch_psnr:.2f}, Time: {time_out:.2f}s, LR: {current_lr:.8f}')

    scheduler_warmup.step()
    logger.debug(f"Learning rate updated to {current_lr:.8f} after scheduler step")

    if (epoch + 1) % 50 == 0:
        model_path = os.path.join(opt.checkpoint_dir, f'model_epoch_{epoch + 1}_SRD.pth')
        torch.save(model.state_dict(), model_path)
        logger.info(f"Saved periodic model checkpoint to {model_path}")

torch.save(model.state_dict(), opt.final_model_path)
logger.info(f"Saved final model to {opt.final_model_path}")
logger.info("Training completed successfully ")
logger.info("=" * 80)
logger.info(f"TRAINING SESSION COMPLETED")
logger.info(f"Best PSNR: {best_psnr:.2f} at iteration {best_iter}")
logger.info("=" * 80)