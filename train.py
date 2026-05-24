# 可以从中断的折继续运行
import os
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score

from model import Transformer
from early_stopping import EarlyStopping
from data_loader import data_generator
from config import Config, Path
from mlflow_utils import MLflowTracker
from compatibility import (
    CHECKPOINT_METADATA_FILE,
    GROUP_GRANULARITY,
    SPLIT_METADATA_FILE,
    SPLIT_RANDOM_STATE,
    SPLIT_SHUFFLE,
    assert_checkpoint_compatibility,
    assert_metadata_compatibility,
    assert_runtime_compatibility,
    checkpoint_metadata_path,
    checkpoint_root,
    current_compatibility_metadata,
    fold_dir as get_fold_dir,
    save_checkpoint_metadata,
)


def set_random_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_label_distribution(labels, split_name='dataset'):
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)

    print(f'\n[{split_name}] label distribution:')
    for u, c in zip(unique, counts):
        print(f'  class {u}: {c} ({c / total:.6f})')


def label_distribution(labels):
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    unique, counts = np.unique(labels, return_counts=True)
    return {f'class_{int(u)}': int(c) for u, c in zip(unique, counts)}


def log_config_params(tracker, config):
    tracker.log_params({
        'num_fold': config.num_fold,
        'val_ratio': config.val_ratio,
        'max_folds_to_run': config.max_folds_to_run,
        'num_classes': config.num_classes,
        'num_epochs': config.num_epochs,
        'batch_size': config.batch_size,
        'pad_size': config.pad_size,
        'tf_seq_len': config.tf_seq_len,
        'context_size': config.context_size,
        'left_context': config.left_context,
        'right_context': config.right_context,
        'learning_rate': config.learning_rate,
        'dropout': config.dropout,
        'dim_model': config.dim_model,
        'forward_hidden': config.forward_hidden,
        'fc_hidden': config.fc_hidden,
        'num_head': config.num_head,
        'num_encoder': config.num_encoder,
        'num_encoder_context': config.num_encoder_context,
        'num_encoder_multi': config.num_encoder_multi,
        'model_variant': config.model_variant,
        'checkpoint_root': checkpoint_root(config),
        'normalization_strategy': config.normalization_strategy,
        'data_generator_interface_version': config.data_generator_interface_version,
        'fusion_boundary_shape': config.fusion_boundary_shape,
        'use_positional_encoding': config.use_positional_encoding,
        'mamba_d_state': config.mamba_d_state,
        'mamba_d_conv': config.mamba_d_conv,
        'mamba_expand': config.mamba_expand,
        'label_smoothing': config.label_smoothing,
        'weight_decay': config.weight_decay,
        'grad_clip': config.grad_clip,
        'early_stop_patience': config.early_stop_patience,
        'early_stop_delta': config.early_stop_delta,
        'device': config.device,
    })


def log_fold_artifacts(tracker, fold_dir):
    for filename in os.listdir(fold_dir):
        if filename.endswith(('.pkl', '.npy', '.npz')):
            tracker.log_artifact(os.path.join(fold_dir, filename), artifact_path=os.path.basename(fold_dir))


def split_window_identity(window_meta, indices):
    selected = [window_meta[int(index)] for index in indices]
    sample_ids = np.asarray([meta['sample_id'] for meta in selected], dtype=str)
    subject_ids = np.asarray([meta['subject_id'] for meta in selected], dtype=str)
    center_epoch_indices = np.asarray(
        [int(meta.get('center_epoch_index', meta.get('epoch_index'))) for meta in selected],
        dtype=np.int64,
    )
    labels = np.asarray([int(meta['label']) for meta in selected], dtype=np.int64)
    return sample_ids, subject_ids, center_epoch_indices, labels


def unique_group_ids(groups):
    return np.unique(np.asarray(groups, dtype=str))


def group_ids_from_window_meta(window_meta):
    return unique_group_ids([meta[GROUP_GRANULARITY] for meta in window_meta])


def assert_disjoint_group_ids(group_sets, split_name):
    normalized = {
        name: set(np.asarray(group_ids, dtype=str).tolist())
        for name, group_ids in group_sets.items()
    }
    names = list(normalized)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1:]:
            overlap = sorted(normalized[left_name] & normalized[right_name])
            if overlap:
                raise RuntimeError(
                    f'[ERROR] {split_name} group leakage between {left_name} and {right_name}: '
                    f'{GROUP_GRANULARITY} overlap={overlap[:10]}'
                )


def assert_group_split_integrity(train_idx, test_idx, groups, val_window_meta=None, split_name='split'):
    train_idx = np.asarray(train_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    if np.intersect1d(train_idx, test_idx).size > 0:
        raise RuntimeError(f'[ERROR] {split_name} train/test indices overlap.')
    if not np.array_equal(np.sort(np.concatenate([train_idx, test_idx])), np.arange(len(groups))):
        raise RuntimeError(f'[ERROR] {split_name} train/test indices do not cover the train_test dataset exactly once.')

    groups = np.asarray(groups, dtype=str)
    train_group_ids = unique_group_ids(groups[train_idx])
    test_group_ids = unique_group_ids(groups[test_idx])
    group_sets = {
        'train': train_group_ids,
        'test': test_group_ids,
    }
    if val_window_meta is not None:
        group_sets['validation'] = group_ids_from_window_meta(val_window_meta)
    assert_disjoint_group_ids(group_sets, split_name)
    return train_group_ids, test_group_ids, group_sets.get('validation', np.asarray([], dtype=str))


def assert_model_supports_dataset_shape(dataset, config):
    expected_variant = 'context' if dataset.dim() == 5 else 'legacy' if dataset.dim() == 4 else 'unsupported'
    if config.model_variant != expected_variant:
        raise RuntimeError(
            f'[ERROR] Config model_variant={config.model_variant!r} does not match training dataset rank '
            f'{dataset.dim()} (expected {expected_variant!r}).'
        )
    if dataset.dim() == 5:
        expected_tail = (config.context_size, 3, config.tf_seq_len, config.dim_model)
    elif dataset.dim() == 4:
        expected_tail = (3, config.tf_seq_len, config.dim_model)
    else:
        raise RuntimeError(
            f'[ERROR] Unsupported dataset rank {dataset.dim()} for training: shape={tuple(dataset.shape)}'
        )
    if tuple(dataset.shape[1:]) != expected_tail:
        raise RuntimeError(
            f'[ERROR] Unsupported dataset shape for training: expected [N, {", ".join(map(str, expected_tail))}], '
            f'got {tuple(dataset.shape)}'
        )


def save_split_metadata(
    fold_dir,
    fold,
    config,
    train_idx,
    test_idx,
    dataset,
    labels,
    groups=None,
    window_meta=None,
    val_window_meta=None,
):
    train_group_ids = np.asarray([], dtype=str)
    test_group_ids = np.asarray([], dtype=str)
    validation_group_ids = np.asarray([], dtype=str)
    if groups is not None:
        train_group_ids, test_group_ids, validation_group_ids = assert_group_split_integrity(
            train_idx,
            test_idx,
            groups,
            val_window_meta=val_window_meta,
            split_name=f'fold {fold}',
        )

    metadata = {
        'fold_index': np.array(fold, dtype=np.int64),
        'num_fold': np.array(config.num_fold, dtype=np.int64),
        'val_ratio': np.array(config.val_ratio, dtype=np.float64),
        'random_state': np.array(SPLIT_RANDOM_STATE, dtype=np.int64),
        'shuffle': np.array(SPLIT_SHUFFLE, dtype=np.bool_),
        'train_idx': np.asarray(train_idx, dtype=np.int64),
        'test_idx': np.asarray(test_idx, dtype=np.int64),
        'dataset_shape': np.asarray(dataset.shape, dtype=np.int64),
        'labels_shape': np.asarray(labels.shape, dtype=np.int64),
        'train_test_dataset_shape': np.asarray(dataset.shape, dtype=np.int64),
        'train_test_labels_shape': np.asarray(labels.shape, dtype=np.int64),
    }
    metadata.update(current_compatibility_metadata(config, dataset))

    if groups is not None:
        groups = np.asarray(groups)
        metadata.update({
            'groups_shape': np.asarray(groups.shape, dtype=np.int64),
            'train_groups': groups[np.asarray(train_idx, dtype=np.int64)].astype(str),
            'test_groups': groups[np.asarray(test_idx, dtype=np.int64)].astype(str),
            'train_group_ids': train_group_ids,
            'test_group_ids': test_group_ids,
            'validation_group_ids': validation_group_ids,
            'group_ids_train': train_group_ids,
            'group_ids_test': test_group_ids,
            'group_ids_val': validation_group_ids,
            'group_granularity': np.array(GROUP_GRANULARITY),
            'group_split_enforced': np.array(True, dtype=np.bool_),
        })

    if window_meta is not None:
        train_sample_ids, train_subject_ids, train_center_epoch_indices, train_window_labels = split_window_identity(
            window_meta,
            train_idx
        )
        test_sample_ids, test_subject_ids, test_center_epoch_indices, test_window_labels = split_window_identity(
            window_meta,
            test_idx
        )
        metadata.update({
            'context_size': np.array(config.context_size, dtype=np.int64),
            'left_context': np.array(config.left_context, dtype=np.int64),
            'right_context': np.array(config.right_context, dtype=np.int64),
            'train_sample_ids': train_sample_ids,
            'test_sample_ids': test_sample_ids,
            'train_subject_ids': train_subject_ids,
            'test_subject_ids': test_subject_ids,
            'train_center_epoch_indices': train_center_epoch_indices,
            'test_center_epoch_indices': test_center_epoch_indices,
            'train_window_labels': train_window_labels,
            'test_window_labels': test_window_labels,
        })

    np.savez(
        os.path.join(fold_dir, SPLIT_METADATA_FILE),
        **metadata,
    )


def validate_existing_split_metadata(
    fold_dir,
    fold,
    config,
    train_idx,
    test_idx,
    dataset,
    labels,
    groups=None,
    window_meta=None,
    val_window_meta=None,
):
    metadata_path = os.path.join(fold_dir, SPLIT_METADATA_FILE)
    if not os.path.exists(metadata_path):
        raise RuntimeError(
            f'[ERROR] Existing fold {fold} is missing {SPLIT_METADATA_FILE}. '
            'Refusing to treat it as complete because its test split cannot be verified.'
        )

    metadata = np.load(metadata_path)
    required_keys = {
        'fold_index',
        'num_fold',
        'val_ratio',
        'random_state',
        'shuffle',
        'train_idx',
        'test_idx',
        'dataset_shape',
        'labels_shape',
        'train_test_dataset_shape',
        'train_test_labels_shape',
    }
    if groups is not None:
        required_keys.update({
            'groups_shape',
            'train_groups',
            'test_groups',
            'train_group_ids',
            'test_group_ids',
            'validation_group_ids',
            'group_granularity',
            'group_split_enforced',
            'group_ids_train',
            'group_ids_test',
            'group_ids_val',
        })
    if dataset.dim() == 5:
        if window_meta is None:
            raise RuntimeError(
                f'[ERROR] Existing fold {fold} cannot validate context split identity without window metadata.'
            )
        required_keys.update({
            'context_size',
            'left_context',
            'right_context',
            'train_sample_ids',
            'test_sample_ids',
            'train_subject_ids',
            'test_subject_ids',
            'train_center_epoch_indices',
            'test_center_epoch_indices',
            'train_window_labels',
            'test_window_labels',
        })
    missing_keys = sorted(required_keys - set(metadata.files))
    if missing_keys:
        raise RuntimeError(
            f'[ERROR] Existing fold {fold} split metadata is missing keys: {missing_keys}. '
            'Use a separate Kfold_models directory or regenerate folds.'
        )
    assert_metadata_compatibility(metadata, config, dataset, fold, 'split')

    checks = {
        'fold_index': int(metadata['fold_index']) == fold,
        'num_fold': int(metadata['num_fold']) == config.num_fold,
        'val_ratio': np.isclose(float(metadata['val_ratio']), config.val_ratio),
        'random_state': int(metadata['random_state']) == SPLIT_RANDOM_STATE,
        'shuffle': bool(metadata['shuffle']) == SPLIT_SHUFFLE,
        'train_idx': np.array_equal(metadata['train_idx'], np.asarray(train_idx, dtype=np.int64)),
        'test_idx': np.array_equal(metadata['test_idx'], np.asarray(test_idx, dtype=np.int64)),
        'dataset_shape': np.array_equal(metadata['dataset_shape'], np.asarray(dataset.shape, dtype=np.int64)),
        'labels_shape': np.array_equal(metadata['labels_shape'], np.asarray(labels.shape, dtype=np.int64)),
        'train_test_dataset_shape': np.array_equal(
            metadata['train_test_dataset_shape'],
            np.asarray(dataset.shape, dtype=np.int64)
        ),
        'train_test_labels_shape': np.array_equal(
            metadata['train_test_labels_shape'],
            np.asarray(labels.shape, dtype=np.int64)
        ),
    }
    if groups is not None:
        groups = np.asarray(groups)
        train_group_ids, test_group_ids, validation_group_ids = assert_group_split_integrity(
            train_idx,
            test_idx,
            groups,
            val_window_meta=val_window_meta,
            split_name=f'existing fold {fold}',
        )
        checks.update({
            'groups_shape': np.array_equal(metadata['groups_shape'], np.asarray(groups.shape, dtype=np.int64)),
            'train_groups': np.array_equal(
                metadata['train_groups'],
                groups[np.asarray(train_idx, dtype=np.int64)].astype(str)
            ),
            'test_groups': np.array_equal(
                metadata['test_groups'],
                groups[np.asarray(test_idx, dtype=np.int64)].astype(str)
            ),
            'train_group_ids': np.array_equal(metadata['train_group_ids'].astype(str), train_group_ids),
            'test_group_ids': np.array_equal(metadata['test_group_ids'].astype(str), test_group_ids),
            'validation_group_ids': np.array_equal(
                metadata['validation_group_ids'].astype(str),
                validation_group_ids,
            ),
            'group_ids_train': np.array_equal(metadata['group_ids_train'].astype(str), train_group_ids),
            'group_ids_test': np.array_equal(metadata['group_ids_test'].astype(str), test_group_ids),
            'group_ids_val': np.array_equal(metadata['group_ids_val'].astype(str), validation_group_ids),
            'group_granularity': str(np.asarray(metadata['group_granularity']).item()) == GROUP_GRANULARITY,
            'group_split_enforced': bool(metadata['group_split_enforced']) is True,
        })
    if dataset.dim() == 5:
        train_sample_ids, train_subject_ids, train_center_epoch_indices, train_window_labels = split_window_identity(
            window_meta,
            train_idx
        )
        test_sample_ids, test_subject_ids, test_center_epoch_indices, test_window_labels = split_window_identity(
            window_meta,
            test_idx
        )
        checks.update({
            'context_size': int(metadata['context_size']) == config.context_size,
            'left_context': int(metadata['left_context']) == config.left_context,
            'right_context': int(metadata['right_context']) == config.right_context,
            'train_sample_ids': np.array_equal(metadata['train_sample_ids'].astype(str), train_sample_ids),
            'test_sample_ids': np.array_equal(metadata['test_sample_ids'].astype(str), test_sample_ids),
            'train_subject_ids': np.array_equal(metadata['train_subject_ids'].astype(str), train_subject_ids),
            'test_subject_ids': np.array_equal(metadata['test_subject_ids'].astype(str), test_subject_ids),
            'train_center_epoch_indices': np.array_equal(
                metadata['train_center_epoch_indices'],
                train_center_epoch_indices
            ),
            'test_center_epoch_indices': np.array_equal(
                metadata['test_center_epoch_indices'],
                test_center_epoch_indices
            ),
            'train_window_labels': np.array_equal(metadata['train_window_labels'], train_window_labels),
            'test_window_labels': np.array_equal(metadata['test_window_labels'], test_window_labels),
        })
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            f'[ERROR] Existing fold {fold} split metadata is incompatible with the current split '
            f'({", ".join(failed)} mismatch). Use a separate Kfold_models directory or regenerate folds.'
        )


def validate_existing_checkpoint_metadata(fold_dir, fold, config, dataset):
    split_path = os.path.join(fold_dir, SPLIT_METADATA_FILE)
    checkpoint_path = checkpoint_metadata_path(config, fold)
    if not os.path.exists(checkpoint_path):
        raise RuntimeError(
            f'[ERROR] Existing fold {fold} is missing {CHECKPOINT_METADATA_FILE}. '
            'Refusing to treat legacy or unversioned checkpoint artifacts as compatible.'
        )
    if not os.path.exists(split_path):
        raise RuntimeError(
            f'[ERROR] Existing fold {fold} is missing {SPLIT_METADATA_FILE}. '
            'Checkpoint compatibility cannot be validated without split metadata.'
        )

    split_metadata = np.load(split_path)
    checkpoint_metadata = np.load(checkpoint_path)
    assert_checkpoint_compatibility(checkpoint_metadata, split_metadata, config, dataset, fold)


def build_dataloader(dataset, batch_size, shuffle, num_workers=8):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None
    )


def evaluate(model, loader, criterion, config, split_name='eval', print_distribution=False):
    model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for data, target in loader:
            data = data.to(config.device, non_blocking=True)
            target = target.to(config.device, non_blocking=True).long()

            output = model(data)
            loss = criterion(output, target)

            batch_size = target.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            pred = torch.argmax(output, dim=1)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    avg_loss = total_loss / total_samples
    accuracy = accuracy_score(all_labels, all_preds)

    if print_distribution:
        pred_u, pred_c = np.unique(all_preds, return_counts=True)
        label_u, label_c = np.unique(all_labels, return_counts=True)

        print(f'\n[{split_name}] prediction distribution:')
        for u, c in zip(pred_u, pred_c):
            print(f'  pred class {u}: {c} ({c / len(all_preds):.6f})')

        print(f'[{split_name}] true label distribution:')
        for u, c in zip(label_u, label_c):
            print(f'  true class {u}: {c} ({c / len(all_labels):.6f})')

    return accuracy, avg_loss


def is_fold_finished(fold_dir: str) -> bool:
    """
    判定某个 fold 是否已经完整跑完。
    你当前 trainer 在 fold 结束后会保存这些 npy 文件，
    所以这些文件都存在时，就认为这个 fold 已完成。
    """
    required_files = [
        'train_LOSS.npy',
        'train_ACC.npy',
        'test_LOSS.npy',
        'test_ACC.npy',
        'val_LOSS.npy',
        'val_ACC.npy',
        'model.pkl',
        SPLIT_METADATA_FILE,
        CHECKPOINT_METADATA_FILE,
    ]
    return all(os.path.exists(os.path.join(fold_dir, f)) for f in required_files)


def has_fold_outputs(fold_dir: str) -> bool:
    if not os.path.isdir(fold_dir):
        return False
    return any(
        name.endswith(('.pkl', '.npy', '.npz'))
        for name in os.listdir(fold_dir)
    )


def has_checkpoint_artifacts(fold_dir: str) -> bool:
    if not os.path.isdir(fold_dir):
        return False
    return any(
        (
            name == CHECKPOINT_METADATA_FILE
            or name.endswith('.pkl')
            or name.endswith('.npy')
        )
        for name in os.listdir(fold_dir)
    )


def find_first_unfinished_fold(num_fold: int, root='./Kfold_models') -> int:
    """
    自动找到第一个未完成的 fold。
    如果都完成了，返回 num_fold。
    """
    for fold in range(num_fold):
        fold_dir = os.path.join(root, f'fold{fold}')
        if not is_fold_finished(fold_dir):
            return fold
    return num_fold


def train(save_all_checkpoint=False, start_fold=None):
    config = Config()
    path = Path()
    tracker = MLflowTracker(config)

    print(f'[INFO] device = {config.device}')
    print(f'[INFO] batch_size = {config.batch_size}')
    print(f'[INFO] learning_rate = {config.learning_rate}')
    print(f'[INFO] num_epochs = {config.num_epochs}')
    print(f'[INFO] num_fold = {config.num_fold}')
    print(f'[INFO] val_ratio = {config.val_ratio}')
    print(f'[INFO] model_variant = {config.model_variant}')
    print(f'[INFO] checkpoint_root = {checkpoint_root(config)}')
    print(f'[INFO] normalization_strategy = {config.normalization_strategy}')
    print(f'[INFO] data_generator_interface_version = {config.data_generator_interface_version}')
    print(f'[INFO] fusion_boundary_shape = {config.fusion_boundary_shape}')
    if config.max_folds_to_run is not None:
        print(f'[INFO] max_folds_to_run = {config.max_folds_to_run} (limits newly trained folds only)')

    dataset, labels, groups, window_meta, val_loader, val_window_meta = data_generator(
        path_labels=path.path_labels,
        path_dataset=path.path_TF,
        config=config,
    )

    print(f'[INFO] dataset shape: {dataset.shape}')
    print(f'[INFO] labels shape: {labels.shape}')
    print(f'[INFO] groups shape: {groups.shape}')
    print(f'[INFO] window metadata rows: {len(window_meta)}')
    print(f'[INFO] val window metadata rows: {len(val_window_meta)}')
    print_label_distribution(labels, split_name='full dataset')
    assert_model_supports_dataset_shape(dataset, config)
    assert_runtime_compatibility(config, dataset)

    kf = StratifiedGroupKFold(
        n_splits=config.num_fold,
        shuffle=True,
        random_state=SPLIT_RANDOM_STATE
    )

    # 自动找未完成 fold
    root = checkpoint_root(config)
    auto_start_fold = find_first_unfinished_fold(config.num_fold, root=root)

    if start_fold is None:
        start_fold = auto_start_fold

    print(f'[INFO] resume start fold = {start_fold}')

    with tracker.start_run(run_name=config.mlflow_run_name or 'train') as _:
        log_config_params(tracker, config)
        tracker.log_params({
            'dataset_shape': tuple(dataset.shape),
            'labels_shape': tuple(labels.shape),
            'dataset_size': len(labels),
            'train_test_dataset_shape': tuple(dataset.shape),
            'train_test_labels_shape': tuple(labels.shape),
            'groups_shape': tuple(groups.shape),
            'window_meta_rows': len(window_meta),
            'val_window_meta_rows': len(val_window_meta),
            'val_ratio': config.val_ratio,
            'max_folds_to_run': config.max_folds_to_run,
            'save_all_checkpoint': save_all_checkpoint,
            'start_fold': start_fold,
            'context_size': config.context_size,
            'left_context': config.left_context,
            'right_context': config.right_context,
            'group_granularity': GROUP_GRANULARITY,
            'group_split_enforced': True,
            'model_variant': config.model_variant,
            'checkpoint_root': root,
            'normalization_strategy': config.normalization_strategy,
            'data_generator_interface_version': config.data_generator_interface_version,
            'tf_seq_len': config.tf_seq_len,
            'fusion_boundary_shape': config.fusion_boundary_shape,
            'fusion_boundary_output_shape': tuple((config.tf_seq_len, config.dim_model)),
        })
        tracker.log_params({f'full_distribution_{k}': v for k, v in label_distribution(labels).items()})

        any_fold_trained = False
        newly_trained_folds = 0

        split_input = np.arange(len(labels))
        split_labels = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
        split_groups = np.asarray(groups, dtype=str)

        for fold, (train_idx, test_idx) in enumerate(kf.split(split_input, split_labels, split_groups)):
            fold_dir = get_fold_dir(config, fold)
            os.makedirs(fold_dir, exist_ok=True)

            # 1) 小于 start_fold 的一律跳过
            if fold < start_fold:
                if is_fold_finished(fold_dir):
                    validate_existing_split_metadata(
                        fold_dir,
                        fold,
                        config,
                        train_idx,
                        test_idx,
                        dataset,
                        labels,
                        groups,
                        window_meta,
                        val_window_meta,
                    )
                    validate_existing_checkpoint_metadata(fold_dir, fold, config, dataset)
                print(f'[INFO] Skip fold {fold} (before start_fold={start_fold}).')
                continue

            # 2) 如果该 fold 已完整完成，也跳过
            if is_fold_finished(fold_dir):
                validate_existing_split_metadata(
                    fold_dir,
                    fold,
                    config,
                    train_idx,
                    test_idx,
                    dataset,
                    labels,
                    groups,
                    window_meta,
                    val_window_meta,
                )
                validate_existing_checkpoint_metadata(fold_dir, fold, config, dataset)
                print(f'[INFO] Skip fold {fold} (already finished).')
                continue

            metadata_path = os.path.join(fold_dir, SPLIT_METADATA_FILE)
            if os.path.exists(metadata_path):
                validate_existing_split_metadata(
                    fold_dir,
                    fold,
                    config,
                    train_idx,
                    test_idx,
                    dataset,
                    labels,
                    groups,
                    window_meta,
                    val_window_meta,
                )
                if os.path.exists(os.path.join(fold_dir, CHECKPOINT_METADATA_FILE)):
                    validate_existing_checkpoint_metadata(fold_dir, fold, config, dataset)
                elif has_checkpoint_artifacts(fold_dir):
                    raise RuntimeError(
                        f'[ERROR] Fold {fold} has existing checkpoint artifacts but no {CHECKPOINT_METADATA_FILE}. '
                        'Refusing to continue because legacy or unversioned checkpoints cannot be validated.'
                    )
            elif has_fold_outputs(fold_dir):
                raise RuntimeError(
                    f'[ERROR] Fold {fold} has existing outputs but no {SPLIT_METADATA_FILE}. '
                    'Refusing to continue because the split for existing artifacts cannot be verified.'
                )

            if (
                config.max_folds_to_run is not None
                and newly_trained_folds >= config.max_folds_to_run
            ):
                print(
                    f'[INFO] Reached max_folds_to_run={config.max_folds_to_run}; '
                    'stopping this invocation.'
                )
                break

            any_fold_trained = True

            print('\n' + '-' * 15 + f' > Fold {fold} < ' + '-' * 15)

            train_group_ids, test_group_ids, validation_group_ids = assert_group_split_integrity(
                train_idx,
                test_idx,
                groups,
                val_window_meta=val_window_meta,
                split_name=f'fold {fold}',
            )
            X_train, X_test = dataset[train_idx], dataset[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]
            save_split_metadata(
                fold_dir,
                fold,
                config,
                train_idx,
                test_idx,
                dataset,
                labels,
                groups,
                window_meta,
                val_window_meta,
            )
            checkpoint_metadata_path = save_checkpoint_metadata(
                config,
                dataset,
                fold,
                train_group_ids,
                test_group_ids,
                validation_group_ids,
            )

            print(f'[INFO][fold {fold}] X_train shape = {X_train.shape}, y_train shape = {y_train.shape}')
            print(f'[INFO][fold {fold}] X_test  shape = {X_test.shape}, y_test  shape = {y_test.shape}')
            print(
                f'[INFO][fold {fold}] group-aware split enforced at {GROUP_GRANULARITY} granularity: '
                f'train_groups={len(train_group_ids)}, test_groups={len(test_group_ids)}, '
                f'validation_groups={len(validation_group_ids)}.'
            )

            print_label_distribution(y_train, split_name=f'fold {fold} train')
            print_label_distribution(y_test, split_name=f'fold {fold} test')

            with tracker.start_run(run_name=f'fold{fold}', nested=True) as _:
                tracker.log_params({
                    'fold_index': fold,
                    'train_size': len(train_idx),
                    'test_size': len(test_idx),
                    'val_size': len(val_loader.dataset) if hasattr(val_loader, 'dataset') else None,
                    'x_train_shape': tuple(X_train.shape),
                    'y_train_shape': tuple(y_train.shape),
                    'x_test_shape': tuple(X_test.shape),
                    'y_test_shape': tuple(y_test.shape),
                    'checkpoint_metadata_path': checkpoint_metadata_path,
                    'model_variant': config.model_variant,
                })
                tracker.log_params({f'train_distribution_{k}': v for k, v in label_distribution(y_train).items()})
                tracker.log_params({f'test_distribution_{k}': v for k, v in label_distribution(y_test).items()})

                train_set = TensorDataset(X_train, y_train)
                test_set = TensorDataset(X_test, y_test)

                train_loader = build_dataloader(
                    dataset=train_set,
                    batch_size=config.batch_size,
                    shuffle=True,
                    num_workers=8
                )
                test_loader = build_dataloader(
                    dataset=test_set,
                    batch_size=config.batch_size,
                    shuffle=False,
                    num_workers=8
                )

                model = Transformer(config).to(config.device)
                criterion = nn.CrossEntropyLoss()

                optimizer = optim.AdamW(
                    model.parameters(),
                    lr=config.learning_rate,
                    weight_decay=0.01
                )

                early_stopping = EarlyStopping(
                    patience=12,
                    verbose=True,
                    save_all_checkpoint=save_all_checkpoint
                )

                train_ACC = []
                train_LOSS = []
                test_ACC = []
                test_LOSS = []
                val_ACC = []
                val_LOSS = []
                stopped_epoch = None
                early_stopped = False

                for epoch in range(config.num_epochs):
                    model.train()

                    total_train_loss = 0.0
                    total_train_correct = 0
                    total_train_samples = 0

                    loop = tqdm(train_loader, total=len(train_loader), desc=f'Fold {fold} Epoch {epoch}')

                    for data, target in loop:
                        data = data.to(config.device, non_blocking=True)
                        target = target.to(config.device, non_blocking=True).long()

                        optimizer.zero_grad()
                        output = model(data)
                        loss = criterion(output, target)
                        loss.backward()
                        optimizer.step()

                        pred = torch.argmax(output, dim=1)

                        batch_size = target.size(0)
                        total_train_loss += loss.item() * batch_size
                        total_train_correct += (pred == target).sum().item()
                        total_train_samples += batch_size

                        train_acc_batch = (pred == target).float().mean().item()

                        loop.set_postfix(
                            loss=f'{loss.item():.4f}',
                            train_acc=f'{train_acc_batch:.4f}'
                        )

                    train_loss = total_train_loss / total_train_samples
                    train_acc = total_train_correct / total_train_samples

                    need_print_dist = (epoch < 3) or (epoch % 10 == 0)

                    test_acc, test_loss = evaluate(
                        model=model,
                        loader=test_loader,
                        criterion=criterion,
                        config=config,
                        split_name='test',
                        print_distribution=need_print_dist
                    )
                    val_acc, val_loss = evaluate(
                        model=model,
                        loader=val_loader,
                        criterion=criterion,
                        config=config,
                        split_name='val',
                        print_distribution=need_print_dist
                    )

                    print(
                        f'Epoch: {epoch:3d} | '
                        f'train loss: {train_loss:.4f} | train acc: {train_acc:.4f} | '
                        f'val acc: {val_acc:.4f} | val loss: {val_loss:.4f} | '
                        f'test acc: {test_acc:.4f} | test loss: {test_loss:.4f}'
                    )

                    train_ACC.append(train_acc)
                    train_LOSS.append(train_loss)
                    test_ACC.append(test_acc)
                    test_LOSS.append(test_loss)
                    val_ACC.append(val_acc)
                    val_LOSS.append(val_loss)

                    model_path = os.path.join(fold_dir, f'model_{fold}_epoch{epoch}.pkl')
                    early_stopping(val_acc, model, path=model_path)
                    stopped_epoch = epoch

                    tracker.log_metrics({
                        'train_loss': train_loss,
                        'train_acc': train_acc,
                        'test_loss': test_loss,
                        'test_acc': test_acc,
                        'val_loss': val_loss,
                        'val_acc': val_acc,
                        'learning_rate': optimizer.param_groups[0]['lr'],
                        'best_val_acc_so_far': early_stopping.best_metric,
                        'early_stopping_counter': early_stopping.counter,
                    }, step=epoch)

                    if early_stopping.early_stop:
                        early_stopped = True
                        print(f'[INFO] Early stopping at epoch {epoch}')
                        break

                np.save(os.path.join(fold_dir, 'train_LOSS.npy'), np.array(train_LOSS))
                np.save(os.path.join(fold_dir, 'train_ACC.npy'), np.array(train_ACC))
                np.save(os.path.join(fold_dir, 'test_LOSS.npy'), np.array(test_LOSS))
                np.save(os.path.join(fold_dir, 'test_ACC.npy'), np.array(test_ACC))
                np.save(os.path.join(fold_dir, 'val_LOSS.npy'), np.array(val_LOSS))
                np.save(os.path.join(fold_dir, 'val_ACC.npy'), np.array(val_ACC))

                best_epoch = int(np.argmax(val_ACC)) if val_ACC else -1
                best_val_acc = float(np.max(val_ACC)) if val_ACC else 0.0
                tracker.log_metrics({
                    'best_val_acc': best_val_acc,
                    'best_epoch': best_epoch,
                    'stopped_epoch': stopped_epoch if stopped_epoch is not None else -1,
                    'early_stopped': int(early_stopped),
                    'final_train_loss': train_LOSS[-1] if train_LOSS else 0.0,
                    'final_train_acc': train_ACC[-1] if train_ACC else 0.0,
                    'final_val_loss': val_LOSS[-1] if val_LOSS else 0.0,
                    'final_val_acc': val_ACC[-1] if val_ACC else 0.0,
                    'final_test_loss': test_LOSS[-1] if test_LOSS else 0.0,
                    'final_test_acc': test_ACC[-1] if test_ACC else 0.0,
                })
                log_fold_artifacts(tracker, fold_dir)

                del model
                torch.cuda.empty_cache()
                newly_trained_folds += 1

        if not any_fold_trained:
            print('[INFO] All discovered folds are already finished. Nothing to do.')


if __name__ == '__main__':
    set_random_seed(0)

    # 自动从第一个未完成的 fold 开始
    train(save_all_checkpoint=False, start_fold=None)

    # 如果你想手动指定从 fold4 开始，也可以改成：
    # train(save_all_checkpoint=False, start_fold=4)
