import torch
from tqdm import tqdm, trange
from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torch.cuda.amp import autocast

ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
psnr_metric = PeakSignalNoiseRatio(data_range=1.0).cuda()
def validate(model, val_loader, use_amp=False):
    model.eval()
    ssim_total = 0.0
    psnr_total = 0.0
    with torch.no_grad():
        for A, B, C in tqdm(val_loader, desc="Validating"):
            A = A.cuda()
            B = B.cuda()
            C = C.cuda()
            if use_amp:
                with autocast():
                    outputs = model(A, C)
                outputs = outputs.float()
            else:
                outputs = model(A, C)
            ssim_total += ssim_metric(outputs, B).item() * A.size(0)
            psnr_total += psnr_metric(outputs, B).item() * A.size(0)

    avg_ssim = ssim_total / len(val_loader.dataset)
    avg_psnr = psnr_total / len(val_loader.dataset)
    return avg_ssim, avg_psnr