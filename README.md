# MultiChannelSleepNet

MultiChannelSleepNet is a sleep staging project for Sleep-EDF style multi-channel EEG/EOG data.

The current final model baseline is Cross B. Its implementation is kept in `model.py` and exposed as `Transformer`.

## Files

- `prepare_dataset.py`: reads Sleep-EDF PSG and Hypnogram EDF files, extracts 30-second epochs, and saves raw channel arrays plus labels.
- `preprocess_tf.py`: converts raw EEG/EOG arrays into time-frequency `.npy` features and normalizes each channel.
- `data_loader.py`: loads TF features and labels, builds train/test data tensors, and creates the held-out validation loader.
- `model.py`: Cross B Transformer model baseline.
- `train.py`: runs stratified K-fold training, validation, checkpoint saving, and early stopping.
- `evaluate.py`: loads trained checkpoints and reports fold-level and aggregate evaluation metrics.
- `config.py`: central training parameters and filesystem paths.
- `early_stopping.py`: early stopping helper used during training.

## Recommended Workflow

Run the project scripts in this order:

```bash
python prepare_dataset.py
python preprocess_tf.py
python train.py
python evaluate.py
```

Copy `.env.example` to `.env` for local overrides. These values can be configured without editing `config.py`:

```dotenv
MCSN_NUM_FOLD=10
MCSN_NUM_EPOCHS=45
MCSN_DATA_ROOT=/openbayes/home/MultiChannelSleepNet
MCSN_BATCH_SIZE=512
```

## File Naming

Old entry-point/module names are no longer kept. Use the current filenames above; compatibility wrappers for the previous names are intentionally not provided.
