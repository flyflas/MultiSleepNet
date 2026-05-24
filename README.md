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
MCSN_VAL_RATIO=0.1
MCSN_MAX_FOLDS_TO_RUN=
MCSN_NUM_EPOCHS=45
MCSN_DATA_ROOT=/openbayes/home/MultiChannelSleepNet
MCSN_BATCH_SIZE=512
MCSN_MODEL_VARIANT=context
MCSN_CHECKPOINT_ROOT=./Kfold_models
```

`MCSN_VAL_RATIO` controls the single global held-out validation split made before K-fold training.
`MCSN_NUM_FOLD` only controls the Stratified K-fold count over the remaining train/test data.
For quick runs, keep the formal split stable with `MCSN_NUM_FOLD=10` and set
`MCSN_MAX_FOLDS_TO_RUN=2` or `MCSN_MAX_FOLDS_TO_RUN=3` to train only a few unfinished folds in one invocation.

## Model Variants And Checkpoints

`MCSN_MODEL_VARIANT=context` trains/evaluates the 5-epoch context pipeline with input shape `[N, 5, 3, 29, 128]`.
`MCSN_MODEL_VARIANT=legacy` trains/evaluates the explicit single-epoch baseline with input shape `[N, 3, 29, 128]`.

Use separate invocations for the two workflows:

```bash
MCSN_MODEL_VARIANT=context python train.py
MCSN_MODEL_VARIANT=context python evaluate.py
MCSN_MODEL_VARIANT=legacy python train.py
MCSN_MODEL_VARIANT=legacy python evaluate.py
```

For `legacy`, the default context settings become `MCSN_CONTEXT_SIZE=1`, `MCSN_LEFT_CONTEXT=0`, and
`MCSN_RIGHT_CONTEXT=0`. For `context`, the default settings remain the offline center-label window
`[t-2, t-1, t, t+1, t+2] -> label[t]`.

Checkpoints are separated by variant under `MCSN_CHECKPOINT_ROOT`:

```text
Kfold_models/context/fold0/model.pkl
Kfold_models/legacy/fold0/model.pkl
```

Each fold writes `split_metadata.npz` and `checkpoint_metadata.npz`. Evaluation refuses to mix incompatible artifacts, including mismatched `model_variant`, `context_size`, `input_shape`, `normalization_strategy`, `data_generator_interface_version`, `tf_seq_len`, `fusion_boundary_shape`, or train/test/validation group IDs. If these checks fail, regenerate the fold under the correct variant directory instead of reusing old `Kfold_models/fold*` artifacts.

## File Naming

Old entry-point/module names are no longer kept. Use the current filenames above; compatibility wrappers for the previous names are intentionally not provided.
