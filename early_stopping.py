import numpy as np
import torch
import os


class EarlyStopping:
    """
    Early stop the training if monitored metric doesn't improve after a given patience.

    mode='max': larger is better, e.g. val_acc
    mode='min': smaller is better, e.g. val_loss
    """
    def __init__(
        self,
        patience=15,
        verbose=False,
        delta=0.0,
        trace_func=print,
        save_all_checkpoint=False,
        mode='max',
        monitor_name=None
    ):
        """
        Args:
            patience (int): How long to wait after last time monitored metric improved.
            verbose (bool): If True, prints a message for each improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            trace_func (function): trace print function.
            save_all_checkpoint (bool): If True, save all checkpoints during training.
            mode (str): 'max' for metrics like val_acc, 'min' for metrics like val_loss.
            monitor_name (str): Optional display name, e.g. 'val_acc' or 'val_loss'.
        """
        assert mode in ['max', 'min'], "mode must be 'max' or 'min'"

        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.trace_func = trace_func
        self.save_all_checkpoint = save_all_checkpoint

        self.mode = mode
        self.monitor_name = monitor_name if monitor_name is not None else ('val_acc' if mode == 'max' else 'val_loss')

        self.counter = 0
        self.best_score = None
        self.early_stop = False

        if self.mode == 'max':
            self.best_metric = -np.inf
        else:
            self.best_metric = np.inf

    def __call__(self, metric_value, model, path):
        """
        metric_value:
            val_acc if mode='max'
            val_loss if mode='min'
        """
        if self.mode == 'max':
            score = metric_value
            improved = score > self.best_score + self.delta if self.best_score is not None else True
        else:
            score = -metric_value
            improved = score > self.best_score + self.delta if self.best_score is not None else True

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(metric_value, model, path)

        elif improved:
            self.best_score = score
            self.save_checkpoint(metric_value, model, path)
            self.counter = 0

        else:
            self.counter += 1
            self.trace_func(
                f'EarlyStopping counter: {self.counter} out of {self.patience} '
                f'(best {self.monitor_name}: {self.best_metric:.6f}, current: {metric_value:.6f})'
            )
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, metric_value, model, path):
        """Save model when monitored metric improves."""
        if self.verbose:
            if self.mode == 'max':
                self.trace_func(
                    f'{self.monitor_name} increased ({self.best_metric:.6f} --> {metric_value:.6f}). Saving model ...'
                )
            else:
                self.trace_func(
                    f'{self.monitor_name} decreased ({self.best_metric:.6f} --> {metric_value:.6f}). Saving model ...'
                )

        if not self.save_all_checkpoint:
            path = os.path.join(os.path.dirname(path), 'model.pkl')

        torch.save(model.state_dict(), path)
        self.best_metric = metric_value
