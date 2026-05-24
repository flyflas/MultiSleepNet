import os

from dotenv import load_dotenv


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(PROJECT_ROOT, '.env'), override=False)

import torch


def _get_int_env(name, default):
    value = os.getenv(name, '').strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f'{name} must be an integer, got {value!r}') from exc


def _get_optional_int_env(name, default=None):
    value = os.getenv(name, '').strip()
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f'{name} must be an integer, got {value!r}') from exc
    if parsed <= 0:
        raise ValueError(f'{name} must be a positive integer, got {parsed}')
    return parsed


def _get_float_env(name, default):
    value = os.getenv(name, '').strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f'{name} must be a float, got {value!r}') from exc


def _get_str_env(name, default):
    value = os.getenv(name, '').strip()
    return value if value else default


class Config(object):
    """args in model and trainer"""
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # basic training settings
        self.num_fold = _get_int_env('MCSN_NUM_FOLD', 10)
        self.val_ratio = _get_float_env('MCSN_VAL_RATIO', 0.1)
        if not 0.0 < self.val_ratio < 1.0:
            raise ValueError(f'MCSN_VAL_RATIO must be between 0 and 1, got {self.val_ratio}')
        self.max_folds_to_run = _get_optional_int_env('MCSN_MAX_FOLDS_TO_RUN', None)
        self.num_classes = 5
        self.num_epochs = _get_int_env('MCSN_NUM_EPOCHS', 45)
        self.batch_size = _get_int_env('MCSN_BATCH_SIZE', 512)
        self.pad_size = 29
        self.learning_rate = 5e-5

        # compatibility / artifact routing
        self.model_variant = _get_str_env('MCSN_MODEL_VARIANT', 'context')
        if self.model_variant not in {'context', 'legacy'}:
            raise ValueError(f'MCSN_MODEL_VARIANT must be "context" or "legacy", got {self.model_variant!r}')
        self.checkpoint_root = _get_str_env('MCSN_CHECKPOINT_ROOT', './Kfold_models')
        self.variant_checkpoint_root = os.path.join(self.checkpoint_root, self.model_variant)
        self.normalization_strategy = _get_str_env('MCSN_NORMALIZATION_STRATEGY', 'global_channel_mean_std')
        self.data_generator_interface_version = _get_int_env('MCSN_DATA_GENERATOR_INTERFACE_VERSION', 2)
        self.fusion_boundary_shape = _get_str_env('MCSN_FUSION_BOUNDARY_SHAPE', 'sequence_tf_tokens')

        # model settings
        self.dropout = 0.1
        self.dim_model = 128
        self.tf_seq_len = _get_int_env('MCSN_TF_SEQ_LEN', self.pad_size)
        default_context_size = 1 if self.model_variant == 'legacy' else 5
        default_left_context = 0 if self.model_variant == 'legacy' else 2
        default_right_context = 0 if self.model_variant == 'legacy' else 2
        self.context_size = _get_int_env('MCSN_CONTEXT_SIZE', default_context_size)
        self.left_context = _get_int_env('MCSN_LEFT_CONTEXT', default_left_context)
        self.right_context = _get_int_env('MCSN_RIGHT_CONTEXT', default_right_context)
        if self.tf_seq_len <= 0:
            raise ValueError(f'MCSN_TF_SEQ_LEN must be positive, got {self.tf_seq_len}')
        if self.context_size <= 0:
            raise ValueError(f'MCSN_CONTEXT_SIZE must be positive, got {self.context_size}')
        if self.left_context < 0:
            raise ValueError(f'MCSN_LEFT_CONTEXT must be non-negative, got {self.left_context}')
        if self.right_context < 0:
            raise ValueError(f'MCSN_RIGHT_CONTEXT must be non-negative, got {self.right_context}')
        if self.left_context + 1 + self.right_context != self.context_size:
            raise ValueError(
                'MCSN_CONTEXT_SIZE must equal MCSN_LEFT_CONTEXT + 1 + MCSN_RIGHT_CONTEXT, '
                f'got {self.context_size} != {self.left_context} + 1 + {self.right_context}'
            )
        if self.model_variant == 'legacy' and (
            self.context_size != 1 or self.left_context != 0 or self.right_context != 0
        ):
            raise ValueError(
                'MCSN_MODEL_VARIANT=legacy requires single-epoch context settings: '
                'MCSN_CONTEXT_SIZE=1, MCSN_LEFT_CONTEXT=0, MCSN_RIGHT_CONTEXT=0.'
            )
        self.forward_hidden = 1024
        self.fc_hidden = 1024
        self.num_head = 8
        self.num_encoder = 16
        self.num_encoder_context = _get_int_env('MCSN_NUM_ENCODER_CONTEXT', 1)
        if self.num_encoder_context <= 0:
            raise ValueError(f'MCSN_NUM_ENCODER_CONTEXT must be positive, got {self.num_encoder_context}')
        self.num_encoder_multi = 4

        # mamba settings
        self.use_positional_encoding = False
        self.mamba_d_state = 16
        self.mamba_d_conv = 4
        self.mamba_expand = 2

        # optimization
        self.label_smoothing = 0.0
        self.weight_decay = 0.01
        self.grad_clip = 1.0

        # dataloader
        self.num_workers = 12

        # scheduler / early stop
        self.scheduler_factor = 0.5
        self.scheduler_patience = 3
        self.scheduler_min_lr = 1e-6

        self.early_stop_patience = 12
        self.early_stop_delta = 0.0

        # logging
        self.print_distribution_first_n_epochs = 3
        self.print_distribution_every = 10

        # mlflow
        self.mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI', '').strip()
        self.mlflow_tracking_username = os.getenv('MLFLOW_TRACKING_USERNAME', '').strip()
        self.mlflow_tracking_password = os.getenv('MLFLOW_TRACKING_PASSWORD', '').strip()
        self.mlflow_tracking_token = os.getenv('MLFLOW_TRACKING_TOKEN', '').strip()
        self.mlflow_experiment_name = os.getenv('MLFLOW_EXPERIMENT_NAME', 'MultiChannelSleepNet').strip()
        self.mlflow_run_name = os.getenv('MLFLOW_RUN_NAME', '').strip()


class Path(object):
    """path of files in this project"""
    def __init__(self):
        old_root = os.getenv('MCSN_DATA_ROOT', '').strip() or '/openbayes/home/MultiChannelSleepNet'

        self.path_PSG = os.path.join(old_root, 'dataset/sleepEDF-78/sleep-cassette')
        self.path_hypnogram = os.path.join(old_root, 'dataset/sleepEDF-78/Hypnogram')
        self.path_raw_data = os.path.join(old_root, 'data/sleepEDF-78/data_array/raw_data')
        self.path_labels = os.path.join(old_root, 'data/sleepEDF-78/data_array/raw_data/labels')
        self.path_TF = os.path.join(old_root, 'data/sleepEDF-78/data_array/TF_data')
        self.path_TF_per_file = os.path.join(self.path_TF, 'per_file')
        self.path_TF_metadata = os.path.join(self.path_TF, 'tf_per_file_metadata.csv')
