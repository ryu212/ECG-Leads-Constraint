import h5py
import torch
from torch.utils.data import Dataset


class ECGHDF5Dataset(Dataset):
    def __init__(self, h5_file, ecg_key="tracings", transform=None):
        self.h5_file = h5_file
        self.ecg_key = ecg_key
        self.transform = transform

        # Chỉ mở để lấy thông tin kích thước
        with h5py.File(self.h5_file, "r") as f:
            self.length = f[self.ecg_key].shape[0]

        self.file = None
        self.data = None

    def _open_file(self):
        if self.file is None:
            self.file = h5py.File(self.h5_file, "r")
            self.data = self.file[self.ecg_key]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        self._open_file()
        ecg = torch.from_numpy(self.data[index]).float()
        if self.transform:
            ecg = self.transform(ecg)
        return ecg

    def __del__(self):
        if self.file is not None:
            self.file.close()
