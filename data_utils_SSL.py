import os
import numpy as np
import torch
from torch import Tensor
import librosa
from torch.utils.data import Dataset

def pad(x, max_len=64600):
    """Pads or truncates audio to a fixed length."""
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

class Dataset_SimpleTest(Dataset):
    """Dataset for simple test files."""
    def __init__(self, test_folder):
        """
        Args:
            test_folder (str): Path to the folder containing test files.
        """
        self.test_folder = test_folder
        self.file_list = [
            os.path.join(test_folder, f) for f in os.listdir(test_folder) if f.endswith('.wav')
        ]
        self.cut = 64600  # Take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_path = self.file_list[index]
        utt_id = os.path.basename(file_path)  # Extract filename
        X, _ = librosa.load(file_path, sr=16000)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, utt_id
