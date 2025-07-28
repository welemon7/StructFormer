# coding=gb2312
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, ToPILImage, RandomRotation, RandomCrop, ColorJitter
import warnings
from tqdm import tqdm, trange
import torch
import os
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------------------------------------------------
# options
import argparse
import options
opt = options.Options().init(argparse.ArgumentParser(description='ShadowRemoval')).parse_args()
print(opt)


# ---------------------------------------------------------------------------------------------------------------------
# dataset
from dataset import SRDTestDataset
test_data_dir = opt.test_data_dir
test_dataset = SRDTestDataset(test_data_dir, img_size=opt.img_size, augment=False)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=opt.num_workers,
    pin_memory=True
)


# ---------------------------------------------------------------------------------------------------------------------
# model
from model import StructFormer
model = StructFormer().cuda()
torch.backends.cudnn.benchmark = True
print("\nstart: ")
model.load_state_dict(torch.load(opt.best_model_path))
model.eval()
save_dir = opt.result_dir
os.makedirs(save_dir, exist_ok=True)
ssim_total = 0.0
psnr_total = 0.0
rmse_total = 0.0
ssim_total_s = 0.0
psnr_total_s = 0.0
rmse_total_s = 0.0
ssim_total_ns = 0.0
psnr_total_ns = 0.0
rmse_total_ns = 0.0


# ---------------------------------------------------------------------------------------------------------------------
# test
with torch.no_grad():
    for idx in trange(len(test_dataset)):
        A, B, C = test_dataset[idx]
        A = A.unsqueeze(0).cuda()
        B = B.unsqueeze(0).cuda()
        C = C.unsqueeze(0).cuda()

        output = model(A, C)
        output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        B_np = B.squeeze(0).permute(1, 2, 0).cpu().numpy()
        mask_np = C.squeeze(0).permute(1, 2, 0).cpu().numpy()
        mask_binary = (mask_np > 0.5).astype(np.float32)
        mask_binary = np.squeeze(mask_binary)

        # ===================== All-region =====================
        gray_output = cv2.cvtColor(output_np, cv2.COLOR_RGB2GRAY)   # model_out-gray
        gray_B = cv2.cvtColor(B_np, cv2.COLOR_RGB2GRAY) # GT-gray
        ssim = ssim_loss(gray_output, gray_B, channel_axis=None, data_range=1.0)  # data_range=1.0
        psnr = psnr_loss(B_np, output_np, data_range=1.0)  # data_range=1.0
        lab_output = cv2.cvtColor(output_np, cv2.COLOR_RGB2LAB)     # model_out-lab
        lab_B = cv2.cvtColor(B_np, cv2.COLOR_RGB2LAB)   # GT-lab
        rmse = np.abs(lab_output - lab_B).mean() * 3

        # ===================== shadow =====================
        gray_output_s = gray_output * mask_binary
        gray_B_s = gray_B * mask_binary
        ssim_s = ssim_loss(gray_output_s, gray_B_s, channel_axis=None,
                           data_range=1.0) if mask_binary.sum() > 0 else 0  # data_range=1.0
        output_s = output_np * mask_binary[..., np.newaxis]
        B_s = B_np * mask_binary[..., np.newaxis]
        psnr_s = psnr_loss(B_s, output_s, data_range=1.0) if mask_binary.sum() > 0 else 0  # data_range=1.0
        if mask_binary.sum() > 0:
            lab_diff_s = lab_output - lab_B
            rmse_s = np.abs(lab_diff_s[mask_binary > 0]).sum() / mask_binary.sum()
        else:
            rmse_s = 0

        # ===================== Non-shadow =====================
        mask_ns = 1 - mask_binary
        gray_output_ns = gray_output * mask_ns
        gray_B_ns = gray_B * mask_ns
        ssim_ns = ssim_loss(gray_output_ns, gray_B_ns, channel_axis=None,
                            data_range=1.0) if mask_ns.sum() > 0 else 0  # data_range=1.0

        output_ns = output_np * mask_ns[..., np.newaxis]
        B_ns = B_np * mask_ns[..., np.newaxis]
        psnr_ns = psnr_loss(B_ns, output_ns, data_range=1.0) if mask_ns.sum() > 0 else 0  # data_range=1.0

        if mask_ns.sum() > 0:
            lab_diff_ns = lab_output - lab_B
            rmse_ns = np.abs(lab_diff_ns[mask_ns > 0]).sum() / mask_ns.sum()
        else:
            rmse_ns = 0

        ssim_total += ssim
        psnr_total += psnr
        rmse_total += rmse
        ssim_total_s += ssim_s
        psnr_total_s += psnr_s
        rmse_total_s += rmse_s
        ssim_total_ns += ssim_ns
        psnr_total_ns += psnr_ns
        rmse_total_ns += rmse_ns

        output_img = ToPILImage()(output.squeeze(0).cpu().clamp(0, 1))
        img_name = test_dataset.image_files[idx]
        output_img.save(os.path.join(save_dir, img_name))

avg_ssim = ssim_total / len(test_dataset)
avg_psnr = psnr_total / len(test_dataset)
avg_rmse = rmse_total / len(test_dataset)
avg_ssim_s = ssim_total_s / len(test_dataset)
avg_psnr_s = psnr_total_s / len(test_dataset)
avg_rmse_s = rmse_total_s / len(test_dataset)
avg_ssim_ns = ssim_total_ns / len(test_dataset)
avg_psnr_ns = psnr_total_ns / len(test_dataset)
avg_rmse_ns = rmse_total_ns / len(test_dataset)
print("\nAll-region - SSIM: {:.4f}, PSNR: {:.2f}, RMSE: {:.4f}".format(avg_ssim, avg_psnr, avg_rmse))
print("shadow-region - SSIM: {:.4f}, PSNR: {:.2f}, RMSE: {:.4f}".format(avg_ssim_s, avg_psnr_s, avg_rmse_s))
print("non-shadow-region - SSIM: {:.4f}, PSNR: {:.2f}, RMSE: {:.4f}".format(avg_ssim_ns, avg_psnr_ns, avg_rmse_ns))
print(f"save to  {os.path.abspath(save_dir)}")