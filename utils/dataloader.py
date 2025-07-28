import os
from dataset import SRDTrainDataset, SRDTestDataset

def get_training_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return SRDTrainDataset(rgb_dir, img_options, None)

def get_validation_data(rgb_dir):
    assert os.path.exists(rgb_dir)
    return SRDTestDataset(rgb_dir, None)
