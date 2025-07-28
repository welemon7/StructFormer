import torch

class MixUp_AUG:
    def __init__(self):
        self.dist = torch.distributions.beta.Beta(torch.tensor([1.2]), torch.tensor([1.2]))

    def aug(self, rgb_gt, rgb_noisy, gray_mask):
        bs = rgb_gt.size(0)
        indices = torch.randperm(bs)
        rgb_gt2 = rgb_gt[indices]
        rgb_noisy2 = rgb_noisy[indices]
        gray_mask2 = gray_mask[indices]
        lam = self.dist.rsample((bs,1)).view(-1, 1, 1, 1).cuda()

        rgb_gt = lam * rgb_gt + (1-lam) * rgb_gt2
        rgb_noisy = lam * rgb_noisy + (1-lam) * rgb_noisy2
        gray_mask = lam * gray_mask + (1-lam) * gray_mask2
        return rgb_gt, rgb_noisy, gray_mask