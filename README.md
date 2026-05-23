# MultiChannelSleepNet

MultiChannelSleepNet is a sleep staging project for Sleep-EDF style multi-channel EEG/EOG data.

The current final model uses TF features plus an SSE raw-waveform branch with concat-linear fusion before cross attention. Its implementation is kept in `model.py` and exposed as `Transformer`.

## Files

- `prepare_dataset.py`: reads Sleep-EDF PSG and Hypnogram EDF files, extracts 30-second epochs, and saves raw channel arrays plus labels.
- `preprocess_tf.py`: converts raw EEG/EOG arrays into time-frequency `.npy` features and normalizes each channel.
- `data_loader.py`: loads TF features and labels, builds normalized SSE raw-waveform windows, builds train/test tensors, and creates the held-out validation loader.
- `model.py`: TF + SSE Transformer with per-channel concat-linear fusion before cross attention.
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

Training and evaluation now consume paired tensors: TF features shaped `[N, 3, 29, 128]` and SSE raw-waveform windows shaped `[N, 3, 29, 200]`. SSE windows are derived from the prepared raw epochs using 2-second windows, 1-second overlap, and 100 Hz sampling, then normalized per channel at dataset level.

Copy `.env.example` to `.env` for local overrides. These values can be configured without editing `config.py`:

```dotenv
MCSN_NUM_FOLD=10
MCSN_VAL_RATIO=0.1
MCSN_MAX_FOLDS_TO_RUN=
MCSN_NUM_EPOCHS=45
MCSN_DATA_ROOT=/openbayes/home/MultiChannelSleepNet
MCSN_BATCH_SIZE=512
```

`MCSN_VAL_RATIO` controls the single global held-out validation split made before K-fold training.
`MCSN_NUM_FOLD` only controls the Stratified K-fold count over the remaining train/test data.
For quick runs, keep the formal split stable with `MCSN_NUM_FOLD=10` and set
`MCSN_MAX_FOLDS_TO_RUN=2` or `MCSN_MAX_FOLDS_TO_RUN=3` to train only a few unfinished folds in one invocation.

## File Naming

Old entry-point/module names are no longer kept. Use the current filenames above; compatibility wrappers for the previous names are intentionally not provided.
