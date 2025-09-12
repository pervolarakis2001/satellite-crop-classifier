import random
import numpy as np
import torch
import os

from sklearn.model_selection import KFold
from torch.utils.data import Subset
import numpy as np

class Identity(object):
    def __call__(self, sample):
        return sample
    
class RandomSamplePixels(object):
    """Randomly draw num_pixels from the available pixels in sample.
    If the total number of pixels is less than num_pixels, one arbitrary pixel is repeated.
    The valid_pixels keeps track of true and repeated pixels.

    Args:
        num_pixels (int): Number of pixels to sample.
    """

    def __init__(self, num_pixels):
        self.num_pixels = num_pixels

    def __call__(self, sample):
        pixels = sample['pixels']
        T, C, S = pixels.shape
        if S > self.num_pixels:
            indices = random.sample(range(S), self.num_pixels)
            x = pixels[:, :, indices]
            valid_pixels = np.ones(self.num_pixels)
        elif S < self.num_pixels:
            x = np.zeros((T, C, self.num_pixels))
            x[..., :S] = pixels
            x[..., S:] = np.stack([x[:, :, 0] for _ in range(S, self.num_pixels)], axis=-1)
            valid_pixels = np.array([1 for _ in range(S)] + [0 for _ in range(S, self.num_pixels)])
        else:
            x = pixels
            valid_pixels = np.ones(self.num_pixels)
        # Repeat valid_pixels across time
        valid_pixels = np.repeat(valid_pixels[np.newaxis].astype(np.float32), x.shape[0], axis=0)
        sample['pixels'] = x
        sample['valid_pixels'] = valid_pixels
        return sample
    
class RandomSampleTimeSteps(object):
    def __init__(self, seq_length):
        self.seq_length = seq_length

    def __call__(self, sample):
        if self.seq_length == -1:
            return sample
        pixels, date_positions, valid_pixels = sample['pixels'], sample['positions'], sample['valid_pixels']
        t = pixels.shape[0]
        if t > self.seq_length:
            indices = sorted(random.sample(range(t), self.seq_length))
            sample['pixels'] = pixels[indices]
            sample['positions'] = date_positions[indices]
            sample['valid_pixels'] = valid_pixels[indices]
        elif t < self.seq_length:
            # Pad with zeros (or repeat first/last frame, as needed)
            pad_len = self.seq_length - t
            pad_pixels = np.zeros((pad_len, pixels.shape[1], pixels.shape[2]), dtype=pixels.dtype)
            pad_positions = np.zeros((pad_len,), dtype=date_positions.dtype)
            pad_valid = np.zeros((pad_len, valid_pixels.shape[1]), dtype=valid_pixels.dtype)

            sample['pixels'] = np.concatenate([pixels, pad_pixels], axis=0)
            sample['positions'] = np.concatenate([date_positions, pad_positions], axis=0)
            sample['valid_pixels'] = np.concatenate([valid_pixels, pad_valid], axis=0)
        else:
            # t == seq_length
            pass  # Nothing to do

        return sample
    

class FixedSamplePixels(object):
    def __init__(self, num_pixels):
        self.num_pixels = num_pixels

    def __call__(self, sample):
        pixels = sample['pixels']  # shape: (T, C, S)
        T, C, S = pixels.shape
        if S >= self.num_pixels:
            indices = list(range(self.num_pixels))  # always select first N pixels
            x = pixels[:, :, indices]
            valid_pixels = np.ones(self.num_pixels)
        else:
            x = np.zeros((T, C, self.num_pixels))
            x[..., :S] = pixels
            x[..., S:] = np.stack([x[:, :, 0] for _ in range(S, self.num_pixels)], axis=-1)
            valid_pixels = np.array([1 for _ in range(S)] + [0 for _ in range(S, self.num_pixels)])
        
        valid_pixels = np.repeat(valid_pixels[np.newaxis].astype(np.float32), x.shape[0], axis=0)
        sample['pixels'] = x
        sample['valid_pixels'] = valid_pixels
        return sample


class Normalize(object):
     """
    Apply per-channel z-score normalization: (x - mean) / std
     input sample['pixels'] is shaped (T, C, S)
    """
     def __init__(self, mean, std):
        self.mean = np.array(mean).reshape(1, -1, 1).astype(np.float32)  # (1, C, 1)
        self.std = np.array(std).reshape(1, -1, 1).astype(np.float32)

     def __call__(self, sample):
        pixels = sample['pixels']  # (T, C, S)
        sample['pixels'] = (pixels - self.mean) / self.std
        return sample
     

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample

class ToTensor(object):
    def __call__(self, sample):
        sample['pixels'] = torch.from_numpy(sample['pixels'].astype(np.float32))
        sample['valid_pixels'] = torch.from_numpy(sample['valid_pixels'].astype(np.float32))
        sample['positions'] = torch.from_numpy(sample['positions'].astype(np.long))
        if 'extra' in sample:
            sample['extra'] = torch.from_numpy(sample['extra'].astype(np.float32))
        if isinstance(sample['label'], int):
            sample['label'] = torch.tensor(sample['label']).long()
        return sample     


def compute_mean_std(dataset):
    count = 0
    mean_total = 0.0
    std_total = 0.0

    for i in range(len(dataset)):
        sample = dataset[i]
        pixels = sample["pixels"]  # shape: (T, C, S)
        if not isinstance(pixels, np.ndarray) or pixels.ndim != 3:
            continue

        mean_per_channel = np.mean(pixels, axis=(0, 2))  # (C,)
        std_per_channel = np.std(pixels, axis=(0, 2))    # (C,)

        mean_total += mean_per_channel
        std_total += std_per_channel
        count += 1

    if count == 0:
        raise ValueError("No valid samples found.")

    mean = (mean_total / count).astype(np.float32)
    std = (std_total / count).astype(np.float32)

    return mean.tolist(), std.tolist()


def compute_kfold_splits(dataset, k=5, seed=42):
    """
    Splits dataset into k folds, returning a list of
    (train_indices, val_indices, mean, std) tuples for each fold.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    all_indices = np.arange(len(dataset))
    folds = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(all_indices)):
        train_subset = Subset(dataset, train_idx)
        # print(f"[Fold {fold_idx+1}] Computing mean/std from {len(train_subset)} training samples...")

        mean, std = compute_mean_std(train_subset)
        folds.append((train_idx.tolist(), val_idx.tolist(), mean, std))

    return folds
     


class RearrangeToEncoderInput:
    def __call__(self, sample):
        # Expecting: pixels [T, C, S] → to [S, T, C]
        sample["pixels"] = sample["pixels"].permute(2, 0, 1)  # [S, T, C]
        sample["valid_pixels"] = sample["valid_pixels"].permute(1, 0)  # [S, T]
        return sample

