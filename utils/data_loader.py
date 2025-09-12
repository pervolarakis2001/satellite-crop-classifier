import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import zarr
import pickle as pkl
from collections import Counter
import datetime as dt
from torch.utils.data._utils.collate import default_collate



class PixelSetData(Dataset):
    def __init__(self, dataset_folder, transform=None, min_sample=200, indices=None):
        super().__init__()
        self.folder = dataset_folder
        self.data_folder = os.path.join(self.folder, "data")
        self.meta_folder = os.path.join(self.folder, "meta")
        self.transform = transform

        # Load and filter labels
        label_path = os.path.join(self.meta_folder, "labels.json")
        self.encoded_labels, self.class_to_idx = self.load_and_filter_labels(label_path, min_sample)
        self.classes = list(self.class_to_idx.keys())
        filtered_indices = set(self.encoded_labels.keys())

        # Create dataset and filter corrupted zarr files
        self.samples, self.metadata = self.make_dataset(self.data_folder, self.meta_folder, indices, filtered_indices)
        self.dates = self.metadata["dates"]
        self.date_positions = self.compute_date_positions(self.metadata["start_date"], self.dates)
        self.date_indices = np.arange(len(self.date_positions))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, parcel_idx, label, extra = self.samples[idx]
        pixels = zarr.load(path)  # (T, C, S)

        sample = {
            "index": idx,
            "parcel_index": parcel_idx,
            "pixels": pixels,
            "valid_pixels": np.ones((pixels.shape[0], pixels.shape[-1]), dtype=np.float32),
            "positions": np.array(self.date_positions),
            "extra": np.array(extra),
            "label": label,
        }   

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_labels(self):
        return np.array([x[2] for x in self.samples])

    def load_and_filter_labels(self, label_path, min_sample):
        with open(label_path, "r") as f:
            labels = json.load(f)

        label_counts = Counter(labels.values())
        filtered_labels = {int(k): v for k, v in labels.items() if label_counts[v] >= min_sample}
        
        label_to_index = {label: idx for idx, label in enumerate(sorted(set(filtered_labels.values())))}
        encoded_labels = {k: label_to_index[v] for k, v in filtered_labels.items()}


        return encoded_labels, label_to_index

    def make_dataset(self, data_folder, meta_folder, indices, filtered_indices):
        metadata = pkl.load(open(os.path.join(meta_folder, "metadata.pkl"), "rb"))
        instances = []
        new_parcel_metadata = []

        for parcel_idx, parcel in enumerate(metadata["parcels"]):
            if indices is not None and parcel_idx not in indices:
                continue
            if parcel_idx not in filtered_indices:
                continue

            class_index = self.encoded_labels[parcel_idx]
            parcel_path = os.path.join(data_folder, f"{parcel_idx}.zarr")
            extra = parcel["geometric_features"]

            # Skip corrupted or unreadable zarr files
            try:
                z = zarr.open(parcel_path, mode='r')
                _ = z.shape
            except Exception as e:
                # print(f"[SKIP] Corrupted or unreadable Zarr file: {parcel_path} ({type(e).__name__}: {e})")
                continue

            instances.append((parcel_path, parcel_idx, class_index, extra))
            new_parcel_metadata.append(parcel)

        metadata["parcels"] = new_parcel_metadata
        return instances, metadata

    def compute_date_positions(self, start_date, dates):
        def to_date(d):
            d = str(d)  # convert int to string
            return dt.datetime(int(d[:4]), int(d[4:6]), int(d[6:]))

        return [abs((to_date(date) - to_date(start_date)).days) for date in dates]


    def compute_acquisition_deltas(self, dates):  
        dates = [str(d) for d in dates]
        start = dt.datetime.strptime(dates[0], "%Y%m%d")
        return np.array([(dt.datetime.strptime(d, "%Y%m%d") - start).days for d in dates], dtype=np.float32)



def count_labels(input_path):
    with open(input_path, "r") as f:
        labels = json.load(f)
        
    label_counts = Counter(labels.values())

    return label_counts


def padded_collate_fn(batch):
    """
    Custom collate_fn that pads all 'pixels' and 'mask' tensors in a batch to the maximum number 
    of pixels Nmax so they can be stacked into regular tensors.
    
    Assumes each sample is a dictionary containing:
      - 'pixels': Tensor [S, C, N_i]
      - 'mask':   Tensor [S, N_i]
      - and extra (e.g. 'label', 'positions', 'extra' κ.ά.)
    Returns a dictionary where:
      - 'pixels': Tensor [B, S, C, N_max]
      - 'mask':   Tensor [B, S, N_max]
      - all other fields are collated using the default collate_fn into standard tensors.
    """
    # Seek S, C and Ni each sample
    all_pixels = [sample['pixels'] for sample in batch]
    all_masks  = [sample['valid_pixels']  for sample in batch]
    
    # S, C same for eaxh parcel and contain  S and C)
    S, C, _ = all_pixels[0].shape
    
    # Find max N_i in batch
    Ns = [pix.shape[2] for pix in all_pixels]
    N_max = max(Ns)
    
    # 2) Zero-padding for each  sample in N_max pixels
    padded_pixels = []
    padded_masks  = []
    for pix, m in zip(all_pixels, all_masks):
        S_i, C_i, N_i = pix.shape  # S_i==S, C_i==C
        # Dimenions for  padding: pad_size = N_max - N_i
        pad_size = N_max - N_i
        
        # α) pad pixels: from [S, C, N_i] → [S, C, N_max]
        if pad_size > 0:
            # Create tensor zero in the end 
            pad_pix = torch.zeros((S, C, pad_size), dtype=pix.dtype, device=pix.device)
            pix_padded = torch.cat([pix, pad_pix], dim=2)
        else:
            pix_padded = pix
        padded_pixels.append(pix_padded)
        
        #pad mask: from [S, N_i] → [S, N_max] (the pad = 0 means “invalid pixel”)
        if pad_size > 0:
            pad_m = torch.zeros((S, pad_size), dtype=m.dtype, device=m.device)
            m_padded = torch.cat([m, pad_m], dim=1)
        else:
            m_padded = m
        padded_masks.append(m_padded)
    
    # stack  padded tensors:
    batch_pixels = torch.stack(padded_pixels, dim=0)  # [B, S, C, N_max]
    batch_masks  = torch.stack(padded_masks,  dim=0)  # [B, S, N_max]
    
    #  Leveaging 'label', 'positions', 'extra' with default_collate
    batch_out = {
        'pixels': batch_pixels,
        'valid_pixels':   batch_masks
    }
    for key in batch[0].keys():
        if key not in ['pixels', 'valid_pixels']:
            batch_out[key] = default_collate([sample[key] for sample in batch])
    
    return batch_out
