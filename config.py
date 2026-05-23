import os

from dotenv import load_dotenv


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(PROJECT_ROOT, '.env'), override=False)

import torch


class Config(object):
    """args in model and trainer"""
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # basic training settings
        self.num_fold = 10
        self.num_classes = 5
        self.num_epochs = 45
        self.batch_size = 512
        self.pad_size = 29
        self.learning_rate = 5e-5

        # model settings
        self.dropout = 0.1
        self.dim_model = 128
        self.forward_hidden = 1024
        self.fc_hidden = 1024
        self.num_head = 8
        self.num_encoder = 16
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
        old_root = '/openbayes/home/MultiChannelSleepNet'

        self.path_PSG = os.path.join(old_root, 'dataset/sleepEDF-78/sleep-cassette')
        self.path_hypnogram = os.path.join(old_root, 'dataset/sleepEDF-78/Hypnogram')
        self.path_raw_data = os.path.join(old_root, 'data/sleepEDF-78/data_array/raw_data')
        self.path_labels = os.path.join(old_root, 'data/sleepEDF-78/data_array/raw_data/labels')
        self.path_TF = os.path.join(old_root, 'data/sleepEDF-78/data_array/TF_data')
