import argparse
import sys
from pathlib import Path
import tensorflow as tf
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from sampling_rate_transform import ResampleTransform

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from hdf5_dataset import ECGHDF5Dataset
import importlib
from project_paths import artifact_path, relative_path, require_repo_root


def get_version_modules(version):
    supported = {"v0", "v1", "v2", "v3", "v4", "v7", "v8", "v12", "v13", "v14", "v15"}
    if version not in supported:
        raise ValueError(f"Unsupported version: {version}")
    module = importlib.import_module(f"{version}.utils_MCMA")

    return (
        module.Trainer,
        module.make_dataloader,
        module.ecg_to_channels_first,
        module.paddingecg,
    )
def decode_tfrecords4c(example, ecglen):
    feature_description = {
        "ecg": tf.io.FixedLenFeature([], tf.string),
    }
    feature_dict = tf.io.parse_single_example(example, feature_description)
    ecg = tf.io.decode_raw(feature_dict["ecg"], out_type=tf.float32)
    ecg = tf.reshape(ecg, [ecglen, 12])
    return ecg

def read_tfrecords(tfrecord_file, ecglen):
    dataset = tf.data.TFRecordDataset(str(tfrecord_file))
    dataset = dataset.map(
        lambda example: decode_tfrecords4c(example, ecglen),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    return dataset


def _count_tfrecords(data_path):
    count = 0
    for _ in tf.data.TFRecordDataset(str(data_path)):
        count += 1
    return count


def _make_tfrecord_loader(data_path, batch_size, ecglen):
    num_samples = _count_tfrecords(data_path)
    dataset = read_tfrecords(data_path, ecglen)
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset, num_samples


def _make_loader(data_path, batch_size, ecglen, make_dataloader):
    data_path = Path(data_path)
    if data_path.suffix == ".pt":
        loader = make_dataloader(
            data_path,
            batch_size=batch_size,
            shuffle=False,
        )
        return loader, len(loader.dataset)

    elif data_path.suffix == ".npy":
        data = np.load(data_path, allow_pickle=False)

        if data.ndim == 2:
            data = data[None, ...]

        if data.ndim != 3:
            raise ValueError(
                f"Expected .npy shape (N, 1024, 12) or (N, 12, 1024), got {data.shape}"
            )

        # (N, 12, L) -> (N, L, 12)
        if data.shape[1] == 12:
            data = np.transpose(data, (0, 2, 1))

        dataset = TensorDataset(torch.from_numpy(data).float())
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        return loader, len(dataset)

    elif data_path.suffix in {".h5", ".hdf5"}:
        dataset = ECGHDF5Dataset(data_path)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        return loader, len(dataset)

    elif data_path.suffix in {".tfrecord", ".tfrecords"} or data_path.suffix == "":
        return _make_tfrecord_loader(
            data_path,
            batch_size,
            ecglen,
        )

    else:
        raise ValueError(f"Unsupported data file: {data_path}")

def _get_batch(batch):
    if hasattr(batch, "numpy"):
        return torch.from_numpy(batch.numpy()).float()
    if isinstance(batch, (list, tuple)):
        return batch[0]
    return batch


def save_generated_ecg(
    data_path,
    modelpath,
    output_file,
    model_name="MCMA",
    lead=0,
    kernel_size=5,
    ecglen=1024,
    segment_len=None,
    pool_size=2,
    batch_size=256,
    device=None,
    resample=False, 
    version="v1",
    restore_input_lead=True,
):
    if lead not in range(12):
        raise ValueError(f"Expected lead index in [0, 11], got {lead}")
    Trainer, make_dataloader, ecg_to_channels_first, paddingecg = get_version_modules(version)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    loader, num_samples = _make_loader(data_path, batch_size, ecglen, make_dataloader)

    segment_len = segment_len or ecglen

    if segment_len % ecglen != 0:
        raise ValueError(
            f"segment_len ({segment_len}) must be divisible by ecglen ({ecglen})"
        )
    
    if model_name == "MCMA":
        trainer = Trainer(
            modelpath=modelpath,
            ecglen=ecglen,
            kernel_size=kernel_size,
            pool_size=pool_size,
            device=device,
        )
        trainer.load(modelpath)
        model = trainer.model.to(device)
        model.eval()
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    generated = np.lib.format.open_memmap(
        output_file,
        mode="w+",
        dtype=np.float32,
        shape=(num_samples, segment_len, 12),
    )

    start = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="generate", dynamic_ncols=True):

            ecg12 = _get_batch(batch)
            B = ecg12.shape[0]

            # (B, segment_len, 12)
            if ecg12.ndim != 3:
                raise ValueError(f"Expected (B,L,12), got {ecg12.shape}")

            if ecg12.shape[1] != segment_len:
                raise ValueError(
                    f"Expected segment_len={segment_len}, got {ecg12.shape[1]}"
                )
            if restore_input_lead:
                # Preserve the measured lead before resampling and segmentation.
                original_lead = ecg12[:, :, lead].detach().cpu().numpy().copy()
            if Path(data_path).suffix in {".h5", ".hdf5"} and resample:
                ecg12 = ResampleTransform(
                    orig_fs=400,
                    target_fs=500,
                )(ecg12.numpy())
                ecg12 = torch.from_numpy(ecg12)
            # (B,L,12) -> (B*n_seg,1024,12)
            ecg12 = ecg12.reshape(-1, ecglen, 12)
            ecg12 = ecg_to_channels_first(ecg12).to(device)

            lead_index = torch.full(
                (ecg12.size(0),),
                lead,
                dtype=torch.long,
                device=device,
            )

            ecg1 = paddingecg(ecg12, lead_index)
            if version in {"v8", "v12", "v13", "v14", "v15"}:
                gen_ecg12, _ = model(ecg1)   
            else:
                gen_ecg12 = model(ecg1)

            # (B*n_seg,12,1024) -> (B*n_seg,1024,12)
            gen_np = (
                gen_ecg12.detach()
                .cpu()
                .permute(0, 2, 1)
                .numpy()
                .astype(np.float32)
            )

            # (B*n_seg,1024,12) -> (B,segment_len,12)
            segment_len_resamling = segment_len
            if Path(data_path).suffix in {".h5", ".hdf5"} and resample:
                segment_len_resamling = int(round(segment_len * 500 / 400))
            gen_np = gen_np.reshape(B, segment_len_resamling, 12)
            if Path(data_path).suffix in {".h5", ".hdf5"} and resample:
                gen_np = ResampleTransform(
                    orig_fs=500,
                    target_fs=400,
                )(gen_np)
            if restore_input_lead:
                gen_np[:, :, lead] = original_lead
            end = start + B
            generated[start:end] = gen_np
            start = end
    generated.flush()
    if start != num_samples:
        raise RuntimeError(
            f"Generated {start} records but allocated output for {num_samples}"
        )
def parse_args():
    parser = argparse.ArgumentParser(description="Save generated 12-lead ECG as .npy.")
    parser.add_argument("--model_name", default="MCMA", help="Currently supports MCMA only.")
    parser.add_argument("--data_path", type=relative_path, required=True)
    parser.add_argument("--modelpath", type=relative_path, required=True)
    parser.add_argument("--output_file", type=artifact_path, required=True)
    parser.add_argument("--lead", default=0, type=int, choices=range(12))
    parser.add_argument(
        "--restore-input-lead",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restore the original input lead after output resampling (default: enabled).",
    )
    parser.add_argument("--kernel_size", default=5, type=int)
    parser.add_argument("--ecglen", default=1024, type=int)
    parser.add_argument("--pool_size", default=2, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--segment_len", default=None, type=int)
    parser.add_argument(
        "--resampling",
        action="store_true",
        help="Enable 400Hz <-> 500Hz resampling for HDF5 data.",
    )
    parser.add_argument(
        "--version",
        default="v1",
        choices=["v0", "v1", "v2", "v3", "v4", "v7", "v8", "v12", "v13", "v14", "v15"],
        help="Model implementation version.",
    )
    return parser.parse_args()


def main():
    require_repo_root()
    args = parse_args()
    save_generated_ecg(
        model_name=args.model_name,
        data_path=args.data_path,
        modelpath=args.modelpath,
        output_file=args.output_file,
        lead=args.lead,
        kernel_size=args.kernel_size,
        ecglen=args.ecglen,
        pool_size=args.pool_size,
        batch_size=args.batch_size,
        segment_len=args.segment_len,
        resample=args.resampling,
        version=args.version,
        restore_input_lead=args.restore_input_lead,
    )
    print("Saved:", args.output_file)
if __name__ == "__main__":
    main()
