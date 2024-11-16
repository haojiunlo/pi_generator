import numpy as np
import torch
from torch.utils.data import Dataset


# Dataset class to handle the pi points
class PiPointDataset(Dataset):
    def __init__(self, xs, ys, image_array):
        self.points = []
        # Get RGB values for each point
        rgb_values = image_array[xs, ys]
        # Combine coordinates and colors
        self.points = np.column_stack([xs, ys, rgb_values])
        self.points = torch.FloatTensor(self.points)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        return self.points[idx]
