import numpy as np
from scipy.signal import resample


class ResampleTransform:
    def __init__(self, orig_fs=400, target_fs=500):
        self.orig_fs = orig_fs
        self.target_fs = target_fs

    def __call__(self, ecg):
        """
        ecg:
            (T, C)     hoặc
            (B, T, C)
        """

        if ecg.ndim == 2:
            time_axis = 0
        elif ecg.ndim == 3:
            time_axis = 1
        else:
            raise ValueError(
                f"Expected ECG shape (T,C) or (B,T,C), got {ecg.shape}"
            )

        old_len = ecg.shape[time_axis]
        new_len = int(round(old_len * self.target_fs / self.orig_fs))

        ecg = resample(ecg, new_len, axis=time_axis)

        return ecg.astype(np.float32)