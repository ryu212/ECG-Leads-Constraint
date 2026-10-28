# How to run

```powershell
# Install the additional dependencies listed in requirements.txt.
python -m pip install -r requirements.txt

# Train the V13 model on PTB-XL with seed 0 and save the checkpoint.
python v13/training_code/train_MCMA_different_kernel_size.py --trainpath datasets/pt/ptbxl_trainset.pt --valpath datasets/pt/ptbxl_valset.pt --modelpath artifacts/checkpoints/v13/Autoencoder_ks_5.pt --epochs 100 --batch_size 256 --lr 0.001 --stage1_epochs 50 --alpha 0.5 --seed 0

# Evaluate reconstructed ECG signals using MSE and PCC on PTB-XL and CPSC2018.
python v13/evaluating_code/evaluate_signal_MCMA.py --modelpath artifacts/checkpoints/v13/Autoencoder_ks_5.pt --testpath datasets/pt/ptbxl_testset.pt --test2path datasets/pt/cpsc2018.pt

# Evaluate heart rate consistency between original and reconstructed ECG signals on PTB-XL.
python v13/evaluating_code/evaluate_model_HR.py --modelpath artifacts/checkpoints/v13/Autoencoder_ks_5.pt --testpath datasets/pt/ptbxl_testset.pt --resultpath artifacts/metrics/v13/hr --fs 500

# Reconstruct 12-lead ECG signals from lead II of CODE-test using V13 and save them as a NumPy array.
python save_genecg.py --version v13 --modelpath artifacts/checkpoints/v13/Autoencoder_ks_5.pt --data_path external/code_test/ecg_tracings.hdf5 --output_file artifacts/generated_ecg/v13_leadII.npy --ecglen 1024 --segment_len 4096 --resampling --lead 1

# Convert reconstructed ECG signals from NumPy to HDF5 with a tracings dataset for the classifier.
python paper_v2/npy_to_hdf5.py --inp artifacts/generated_ecg/v13_leadII.npy --out artifacts/generated_ecg/v13_leadII.hdf5

# Run the classifier by Ribeiro et al. on reconstructed ECG signals and save the predicted disease probabilities.
python external/automatic-ecg-diagnosis/predict.py artifacts/generated_ecg/v13_leadII.hdf5 external/model.hdf5 --output_file artifacts/generated_ecg/v13_leadII_predictions.npy

# Evaluate diagnostic predictions using precision, recall, F1 and AUPRC against the CODE-test reference labels.
python paper_v2/eval_diagnostic.py --pred artifacts/generated_ecg/v13_leadII_predictions.npy --gold external/code_test/gold_standard.csv --tag v13_leadII --out artifacts/metrics/E2/v13_leadII.csv
```

References and acknowledgments:

- [CHENJIAR3/MCMA](https://github.com/CHENJIAR3/MCMA) — original MCMA implementation by Jiarong Chen et al. for reconstructing 12-lead ECG signals from an arbitrary single lead.
- [antonior92/automatic-ecg-diagnosis](https://github.com/antonior92/automatic-ecg-diagnosis) — ECG classifier implementation by Antônio H. Ribeiro et al., accompanying *Automatic diagnosis of the 12-lead ECG using a deep neural network*.
