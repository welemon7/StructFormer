import os
import torch
import argparse
class Options():
    """docstring for Options"""

    def __init__(self):
        pass

    def init(self,parser):
        # global settings
        parser.add_argument('--batch_size', type=int, default=8, help='batch size')
        parser.add_argument('--epoch', type=int, default=600, help='training epochs')
        parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rating')
        parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
        parser.add_argument('--num_workers', type=int, default=4, help='num workers')
        parser.add_argument('--use_cuda', type=bool, default=True, help='cuda using')

        # args for saving
        parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_SRD', help='save all model')
        parser.add_argument('--best_model_path', type=str, default='./checkpoints_SRD/model_best_SRD.pth', help='best model in training')
        parser.add_argument('--final_model_path', type=str, default='./checkpoints_SRD/model_final_SRD.pth',
                            help='final model in training')
        parser.add_argument('--result_dir', type=str, default='./result_SRD', help='image save')
        parser.add_argument('--log_dir', type=str, default='./checkpoints_SRD/logs', help='logs save')

        # args for training
        parser.add_argument('--augment', type=bool, default=True, help='training augment')
        parser.add_argument('--img_size', type=int, default=256, help='input image size')
        parser.add_argument('--train_data_dir', type=str, default='/media/luosihui/Dataset/SRD/train/', help='training dataset path')
        parser.add_argument('--test_data_dir', type=str, default='/media/luosihui/Dataset/SRD/test/', help='testing dataset path')
        parser.add_argument('--validation_freq', type=int, default=1000, help='validation frequency in iterations')
        parser.add_argument('--warmup_epochs', type=int, default=10, help='warmup scheduler epochs')
        parser.add_argument('--use_mixup', action='store_true', default=False, help='mixup open or close')
        parser.add_argument('--mixup_start_epoch', type=int, default=10, help='epoch to start mixup augmentation')
        parser.add_argument('--use_amp', action='store_true', default=False, help='whether to use mixed precision training')


        # args for seed
        parser.add_argument('--seed', type=int, default=42,
                            help='Random seed for reproducibility (default: 42)')
        parser.add_argument('--deterministic', action='store_true',
                            help='Enable deterministic mode for full reproducibility')
        parser.add_argument('--cudnn-benchmark', action='store_true',
                            help='Enable cuDNN benchmark for potentially faster training')

        return parser







