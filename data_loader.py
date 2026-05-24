import os
import csv
import re
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit

from config import Config
from compatibility import (
    DATA_GENERATOR_INTERFACE_VERSION,
    GROUP_GRANULARITY,
    MODEL_VARIANT_CONTEXT,
    MODEL_VARIANT_LEGACY,
    SPLIT_RANDOM_STATE,
)


TF_METADATA_FILE_NAME = 'tf_per_file_metadata.csv'
TF_METADATA_FIELDS = [
    'sample_id',
    'subject_id',
    'num_epochs',
    'EEG Fpz-Cz TF path',
    'EEG Pz-Oz TF path',
    'EOG TF path',
    'label path',
]
SLEEP_EDF_SUBJECT_RE = re.compile(r'^[A-Za-z]{2}\d{4}')


def infer_subject_id_from_sample_id(sample_id):
    """Infer the expected Sleep-EDF subject id from a metadata sample id."""
    match = SLEEP_EDF_SUBJECT_RE.match(sample_id)
    if match:
        return match.group(0)
    return sample_id[:6] if len(sample_id) >= 6 else sample_id


def validate_metadata_path_sample_id(row_index, sample_id, relative_path, expected_suffix):
    """Ensure metadata path basename points to the same sample id."""
    basename = os.path.basename(relative_path)
    if not basename.endswith(expected_suffix):
        raise ValueError(
            f'[ERROR] Row {row_index} path has unexpected suffix for {sample_id}: {relative_path}'
        )

    path_sample_id = basename[:-len(expected_suffix)]
    if path_sample_id != sample_id:
        raise ValueError(
            f'[ERROR] Row {row_index} sample_id/path mismatch: '
            f'sample_id={sample_id}, path sample_id={path_sample_id}, path={relative_path}'
        )


def print_label_distribution(labels, split_name='labels'):
    """Print label distribution."""
    if isinstance(labels, torch.Tensor):
        labels_np = labels.cpu().numpy()
    else:
        labels_np = labels

    unique, counts = np.unique(labels_np, return_counts=True)
    total = len(labels_np)

    print(f'[{split_name}] label distribution:')
    if total == 0:
        print('  empty')
        return
    for u, c in zip(unique, counts):
        print(f'  class {u}: {c} ({c / total:.6f})')


def assert_data_generator_contract(config):
    if config.data_generator_interface_version != DATA_GENERATOR_INTERFACE_VERSION:
        raise ValueError(
            '[ERROR] Unsupported data_generator interface version: '
            f'config={config.data_generator_interface_version}, expected={DATA_GENERATOR_INTERFACE_VERSION}.'
        )
    if config.model_variant not in {MODEL_VARIANT_CONTEXT, MODEL_VARIANT_LEGACY}:
        raise ValueError(f'[ERROR] Unsupported model_variant for data_generator: {config.model_variant!r}')


def assert_context_dataset_shape(dataset, labels, config, split_name='context dataset'):
    """Fail fast if a context dataset is not shaped as [N, C, 3, T, D]."""
    expected_tail = (config.context_size, 3, config.tf_seq_len, config.dim_model)
    if tuple(dataset.shape[1:]) != expected_tail:
        raise ValueError(
            f'[ERROR] {split_name} has invalid context shape: '
            f'expected [N, {", ".join(map(str, expected_tail))}], got {tuple(dataset.shape)}'
        )
    if len(dataset) != len(labels):
        raise ValueError(
            f'[ERROR] {split_name} dataset size ({len(dataset)}) != labels size ({len(labels)})'
        )


def assert_legacy_dataset_shape(dataset, labels, config, split_name='legacy dataset'):
    """Fail fast if a legacy dataset is not shaped as [N, 3, T, D]."""
    expected_tail = (3, config.tf_seq_len, config.dim_model)
    if tuple(dataset.shape[1:]) != expected_tail:
        raise ValueError(
            f'[ERROR] {split_name} has invalid legacy shape: '
            f'expected [N, {", ".join(map(str, expected_tail))}], got {tuple(dataset.shape)}'
        )
    if len(dataset) != len(labels):
        raise ValueError(
            f'[ERROR] {split_name} dataset size ({len(dataset)}) != labels size ({len(labels)})'
        )


def assert_disjoint_groups(train_groups, val_groups, split_name='validation'):
    """Ensure no subject-level group appears on both sides of a split."""
    train_group_ids = set(np.asarray(train_groups, dtype=str).tolist())
    val_group_ids = set(np.asarray(val_groups, dtype=str).tolist())
    overlap = sorted(train_group_ids & val_group_ids)
    if overlap:
        raise ValueError(
            f'[ERROR] {split_name} group split leakage: '
            f'{GROUP_GRANULARITY} overlap between train/test and validation: {overlap[:10]}'
        )


def select_window_meta(window_meta, indices):
    return [window_meta[int(index)] for index in indices]


def group_train_validation_split(dataset, labels, groups, window_meta, val_ratio):
    """Split context windows by subject so all windows for one subject stay together."""
    group_array = np.asarray(groups, dtype=object)
    unique_groups = np.unique(group_array.astype(str))
    if len(unique_groups) < 2:
        raise ValueError(
            f'[ERROR] Need at least 2 unique {GROUP_GRANULARITY} groups for held-out validation, '
            f'got {len(unique_groups)}.'
        )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=val_ratio,
        random_state=SPLIT_RANDOM_STATE,
    )
    split_input = np.arange(len(labels))
    train_idx, val_idx = next(splitter.split(split_input, labels.cpu().numpy(), group_array.astype(str)))

    groups_train_test = group_array[train_idx]
    groups_val = group_array[val_idx]
    assert_disjoint_groups(groups_train_test, groups_val)

    meta_train_test = select_window_meta(window_meta, train_idx)
    meta_val = select_window_meta(window_meta, val_idx)

    return (
        dataset[train_idx],
        dataset[val_idx],
        labels[train_idx],
        labels[val_idx],
        groups_train_test,
        groups_val,
        meta_train_test,
        meta_val,
    )


def build_context_windows_for_file(path_dataset, row, config):
    """Build [N, context_size, 3, tf_seq_len, dim_model] windows without crossing file boundaries."""
    sample_id = row['sample_id']
    subject_id = row['subject_id']
    num_epochs = int(row['num_epochs'])

    channel_arrays = [
        np.load(os.path.join(path_dataset, row['EEG Fpz-Cz TF path'])).astype('float32', copy=False),
        np.load(os.path.join(path_dataset, row['EEG Pz-Oz TF path'])).astype('float32', copy=False),
        np.load(os.path.join(path_dataset, row['EOG TF path'])).astype('float32', copy=False),
    ]
    labels = np.load(os.path.join(path_dataset, row['label path'])).astype(np.int64, copy=False)

    per_file_dataset = np.stack(channel_arrays, axis=1)
    expected_file_shape = (num_epochs, 3, config.tf_seq_len, config.dim_model)
    if per_file_dataset.shape != expected_file_shape:
        raise ValueError(
            f'[ERROR] {sample_id} stacked TF shape mismatch: '
            f'expected {expected_file_shape}, got {per_file_dataset.shape}'
        )
    if labels.shape != (num_epochs,):
        raise ValueError(
            f'[ERROR] {sample_id} label shape mismatch: '
            f'expected ({num_epochs},), got {labels.shape}'
        )

    num_windows = max(num_epochs - config.left_context - config.right_context, 0)
    if num_windows == 0:
        empty_windows = np.empty(
            (0, config.context_size, 3, config.tf_seq_len, config.dim_model),
            dtype=np.float32,
        )
        empty_labels = np.empty((0,), dtype=np.int64)
        return empty_windows, empty_labels, [], np.empty((0,), dtype=object)

    windows = np.empty(
        (num_windows, config.context_size, 3, config.tf_seq_len, config.dim_model),
        dtype=np.float32,
    )
    window_labels = np.empty((num_windows,), dtype=np.int64)
    window_meta = []
    groups = np.empty((num_windows,), dtype=object)

    for window_index, center_epoch_index in enumerate(range(config.left_context, num_epochs - config.right_context)):
        start = center_epoch_index - config.left_context
        end = center_epoch_index + config.right_context + 1
        windows[window_index] = per_file_dataset[start:end]
        window_label = int(labels[center_epoch_index])
        window_labels[window_index] = window_label
        groups[window_index] = subject_id
        window_meta.append({
            'sample_id': sample_id,
            'subject_id': subject_id,
            'center_epoch_index': int(center_epoch_index),
            'label': window_label,
        })

    for check_index in sorted(set([0, num_windows // 2, num_windows - 1])):
        meta = window_meta[check_index]
        center_epoch_index = int(meta['center_epoch_index'])
        if not config.left_context <= center_epoch_index < num_epochs - config.right_context:
            raise ValueError(
                f'[ERROR] {sample_id} context center out of bounds at window {check_index}: '
                f'center={center_epoch_index}, num_epochs={num_epochs}'
            )
        if int(window_labels[check_index]) != int(labels[center_epoch_index]):
            raise ValueError(
                f'[ERROR] {sample_id} context label mismatch at window {check_index}: '
                f'window_label={window_labels[check_index]}, source_label={labels[center_epoch_index]}'
            )
        if int(window_labels[check_index]) != int(meta['label']):
            raise ValueError(
                f'[ERROR] {sample_id} context metadata label mismatch at window {check_index}: '
                f'window_label={window_labels[check_index]}, meta={meta}'
            )
        if groups[check_index] != subject_id or meta['sample_id'] != sample_id or meta['subject_id'] != subject_id:
            raise ValueError(f'[ERROR] {sample_id} context metadata identity mismatch at window {check_index}: {meta}')

        start = center_epoch_index - config.left_context
        for context_offset in range(config.context_size):
            source_epoch_index = start + context_offset
            if not np.array_equal(windows[check_index, context_offset], per_file_dataset[source_epoch_index]):
                raise ValueError(
                    f'[ERROR] {sample_id} context window mismatch at window {check_index}, '
                    f'offset={context_offset}, source_epoch={source_epoch_index}'
                )

    return windows, window_labels, window_meta, groups


def build_context_dataset(path_dataset, metadata_rows, config):
    """Build the dense context dataset plus aligned labels, groups, and metadata."""
    datasets = []
    labels = []
    groups = []
    window_meta = []
    expected_windows = 0

    for row in metadata_rows:
        expected_windows += max(int(row['num_epochs']) - config.left_context - config.right_context, 0)
        file_windows, file_labels, file_meta, file_groups = build_context_windows_for_file(path_dataset, row, config)
        if len(file_windows) == 0:
            continue
        datasets.append(file_windows)
        labels.append(file_labels)
        groups.append(file_groups)
        window_meta.extend(file_meta)

    if datasets:
        dataset = np.concatenate(datasets, axis=0)
        label_array = np.concatenate(labels, axis=0)
        group_array = np.concatenate(groups, axis=0)
    else:
        dataset = np.empty((0, config.context_size, 3, config.tf_seq_len, config.dim_model), dtype=np.float32)
        label_array = np.empty((0,), dtype=np.int64)
        group_array = np.empty((0,), dtype=object)

    if len(dataset) != expected_windows:
        raise ValueError(
            f'[ERROR] Context dataset length mismatch: expected {expected_windows}, got {len(dataset)}'
        )
    if len(window_meta) != len(dataset) or len(group_array) != len(dataset):
        raise ValueError(
            '[ERROR] Context metadata/groups are not aligned with context windows: '
            f'windows={len(dataset)}, meta={len(window_meta)}, groups={len(group_array)}'
        )

    if len(dataset) > 0:
        # Spot-check chronological ordering and center labels without relying on random state.
        for check_index in sorted(set([0, len(dataset) // 2, len(dataset) - 1])):
            meta = window_meta[check_index]
            if int(label_array[check_index]) != int(meta['label']):
                raise ValueError(
                    f'[ERROR] Context label mismatch at window {check_index}: '
                    f'label={label_array[check_index]}, meta={meta}'
                )
            if int(meta['center_epoch_index']) < config.left_context:
                raise ValueError(f'[ERROR] Invalid center epoch metadata at window {check_index}: {meta}')

    return dataset, label_array, group_array, window_meta


def build_legacy_epochs_for_file(path_dataset, row, config):
    """Build [E, 3, tf_seq_len, dim_model] single-epoch samples for the explicit legacy variant."""
    sample_id = row['sample_id']
    subject_id = row['subject_id']
    num_epochs = int(row['num_epochs'])
    channel_arrays = [
        np.load(os.path.join(path_dataset, row['EEG Fpz-Cz TF path'])).astype('float32', copy=False),
        np.load(os.path.join(path_dataset, row['EEG Pz-Oz TF path'])).astype('float32', copy=False),
        np.load(os.path.join(path_dataset, row['EOG TF path'])).astype('float32', copy=False),
    ]
    labels = np.load(os.path.join(path_dataset, row['label path'])).astype(np.int64, copy=False)

    per_file_dataset = np.stack(channel_arrays, axis=1)
    expected_file_shape = (num_epochs, 3, config.tf_seq_len, config.dim_model)
    if per_file_dataset.shape != expected_file_shape:
        raise ValueError(
            f'[ERROR] {sample_id} legacy TF shape mismatch: '
            f'expected {expected_file_shape}, got {per_file_dataset.shape}'
        )
    if labels.shape != (num_epochs,):
        raise ValueError(
            f'[ERROR] {sample_id} legacy label shape mismatch: expected ({num_epochs},), got {labels.shape}'
        )

    groups = np.asarray([subject_id] * num_epochs, dtype=object)
    epoch_meta = [
        {
            'sample_id': sample_id,
            'subject_id': subject_id,
            'epoch_index': int(epoch_index),
            'label': int(labels[epoch_index]),
        }
        for epoch_index in range(num_epochs)
    ]
    return per_file_dataset, labels, epoch_meta, groups


def build_legacy_dataset(path_dataset, metadata_rows, config):
    """Build the single-epoch legacy dataset while preserving subject groups."""
    datasets = []
    labels = []
    groups = []
    epoch_meta = []
    expected_epochs = 0

    for row in metadata_rows:
        expected_epochs += int(row['num_epochs'])
        file_dataset, file_labels, file_meta, file_groups = build_legacy_epochs_for_file(path_dataset, row, config)
        datasets.append(file_dataset)
        labels.append(file_labels)
        groups.append(file_groups)
        epoch_meta.extend(file_meta)

    if datasets:
        dataset = np.concatenate(datasets, axis=0)
        label_array = np.concatenate(labels, axis=0)
        group_array = np.concatenate(groups, axis=0)
    else:
        dataset = np.empty((0, 3, config.tf_seq_len, config.dim_model), dtype=np.float32)
        label_array = np.empty((0,), dtype=np.int64)
        group_array = np.empty((0,), dtype=object)

    if len(dataset) != expected_epochs:
        raise ValueError(f'[ERROR] Legacy dataset length mismatch: expected {expected_epochs}, got {len(dataset)}')
    if len(epoch_meta) != len(dataset) or len(group_array) != len(dataset):
        raise ValueError(
            '[ERROR] Legacy metadata/groups are not aligned with single-epoch samples: '
            f'epochs={len(dataset)}, meta={len(epoch_meta)}, groups={len(group_array)}'
        )

    return dataset, label_array, group_array, epoch_meta


def load_per_file_tf_metadata(path_dataset, config):
    """Load and validate per-file TF metadata generated by preprocess_tf.py."""
    metadata_path = os.path.join(path_dataset, TF_METADATA_FILE_NAME)
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f'[ERROR] Missing per-file TF metadata: {metadata_path}')

    rows = []
    with open(metadata_path, newline='') as f:
        reader = csv.DictReader(f)
        missing_fields = sorted(set(TF_METADATA_FIELDS) - set(reader.fieldnames or []))
        if missing_fields:
            raise ValueError(f'[ERROR] Metadata file is missing fields: {missing_fields}')
        rows = list(reader)

    seen_sample_ids = set()
    for row_index, row in enumerate(rows):
        sample_id = row['sample_id'].strip()
        subject_id = row['subject_id'].strip()

        if not sample_id:
            raise ValueError(f'[ERROR] Row {row_index} has empty sample_id.')
        if sample_id in seen_sample_ids:
            raise ValueError(f'[ERROR] Duplicate sample_id in metadata: {sample_id}')
        seen_sample_ids.add(sample_id)

        if not subject_id:
            raise ValueError(f'[ERROR] Row {row_index} sample_id={sample_id} has empty subject_id.')

        expected_subject_id = infer_subject_id_from_sample_id(sample_id)
        if subject_id != expected_subject_id:
            raise ValueError(
                f'[ERROR] Row {row_index} subject_id mismatch for {sample_id}: '
                f'expected {expected_subject_id}, got {subject_id}'
            )

        row['sample_id'] = sample_id
        row['subject_id'] = subject_id
        try:
            num_epochs = int(row['num_epochs'])
        except ValueError as exc:
            raise ValueError(f'[ERROR] Invalid num_epochs for row {row_index}: {row["num_epochs"]!r}') from exc

        tf_paths = [
            row['EEG Fpz-Cz TF path'].strip(),
            row['EEG Pz-Oz TF path'].strip(),
            row['EOG TF path'].strip(),
        ]
        label_path = row['label path'].strip()

        validate_metadata_path_sample_id(row_index, sample_id, tf_paths[0], '_TF_EEG_Fpz-Cz_mean_std.npy')
        validate_metadata_path_sample_id(row_index, sample_id, tf_paths[1], '_TF_EEG_Pz-Oz_mean_std.npy')
        validate_metadata_path_sample_id(row_index, sample_id, tf_paths[2], '_TF_EOG_mean_std.npy')
        validate_metadata_path_sample_id(row_index, sample_id, label_path, '_label.npy')

        for relative_path in tf_paths + [label_path]:
            full_path = os.path.join(path_dataset, relative_path)
            if not os.path.exists(full_path):
                raise FileNotFoundError(f'[ERROR] Metadata path does not exist for {sample_id}: {full_path}')

        channel_shapes = [
            np.load(os.path.join(path_dataset, relative_path), mmap_mode='r').shape
            for relative_path in tf_paths
        ]
        label_shape = np.load(os.path.join(path_dataset, label_path), mmap_mode='r').shape

        expected_tf_shape = (num_epochs, config.tf_seq_len, config.dim_model)
        if any(shape != expected_tf_shape for shape in channel_shapes):
            raise ValueError(
                f'[ERROR] {sample_id} channel shape mismatch: '
                f'expected {expected_tf_shape}, got {channel_shapes}'
            )
        if label_shape != (num_epochs,):
            raise ValueError(
                f'[ERROR] {sample_id} label shape mismatch: '
                f'expected ({num_epochs},), got {label_shape}'
            )

        row['num_epochs'] = num_epochs
        row['EEG Fpz-Cz TF path'] = tf_paths[0]
        row['EEG Pz-Oz TF path'] = tf_paths[1]
        row['EOG TF path'] = tf_paths[2]
        row['label path'] = label_path

    print(f'[INFO] Loaded {len(rows)} per-file TF metadata rows from {metadata_path}')
    return rows


def build_validation_loader(X_val, y_val, config):
    val_set = TensorDataset(X_val, y_val)
    num_workers = int(config.num_workers)
    return DataLoader(
        dataset=val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )


def data_generator(path_labels, path_dataset, config=None):
    config = config or Config()
    assert_data_generator_contract(config)

    if path_labels is not None:
        print('[INFO] path_labels is unused; labels are loaded from per-file TF metadata.')

    metadata_rows = load_per_file_tf_metadata(path_dataset, config)
    if config.model_variant == MODEL_VARIANT_CONTEXT:
        dataset_np, labels_np, groups, sample_meta = build_context_dataset(path_dataset, metadata_rows, config)
        split_name = 'context'
    elif config.model_variant == MODEL_VARIANT_LEGACY:
        dataset_np, labels_np, groups, sample_meta = build_legacy_dataset(path_dataset, metadata_rows, config)
        split_name = 'legacy'
    else:
        raise ValueError(f'[ERROR] Unsupported model_variant: {config.model_variant!r}')

    dataset = torch.from_numpy(dataset_np)
    labels = torch.from_numpy(labels_np)
    if config.model_variant == MODEL_VARIANT_CONTEXT:
        assert_context_dataset_shape(dataset, labels, config, split_name='full context dataset')
    else:
        assert_legacy_dataset_shape(dataset, labels, config, split_name='full legacy dataset')

    print(f'[INFO] data_generator_interface_version: {config.data_generator_interface_version}')
    print(f'[INFO] model_variant: {config.model_variant}')
    print(f'[INFO] {split_name} dataset shape: {dataset.shape}')
    print(f'[INFO] {split_name} labels shape: {labels.shape}')
    print(f'[INFO] {split_name} groups shape: {groups.shape}')
    print(f'[INFO] {split_name} metadata rows: {len(sample_meta)}')
    if config.model_variant == MODEL_VARIANT_CONTEXT:
        expected_context_windows = sum(
            max(int(row['num_epochs']) - config.left_context - config.right_context, 0)
            for row in metadata_rows
        )
        print(f'[INFO] expected context windows: {expected_context_windows}')

    print_label_distribution(labels, split_name=f'full {split_name} dataset')

    X_train_test, X_val, y_train_test, y_val, groups_train_test, groups_val, meta_train_test, meta_val = (
        group_train_validation_split(
        dataset,
        labels,
        groups,
        sample_meta,
        val_ratio=config.val_ratio,
        )
    )

    if config.model_variant == MODEL_VARIANT_CONTEXT:
        assert_context_dataset_shape(X_train_test, y_train_test, config, split_name='train_test context dataset')
        assert_context_dataset_shape(X_val, y_val, config, split_name='val context dataset')
    else:
        assert_legacy_dataset_shape(X_train_test, y_train_test, config, split_name='train_test legacy dataset')
        assert_legacy_dataset_shape(X_val, y_val, config, split_name='val legacy dataset')

    print(f'[INFO] train_test {split_name} size: {len(X_train_test)}')
    print(f'[INFO] val {split_name} size: {len(X_val)}')
    print(f'[INFO] train_test unique groups: {len(set(groups_train_test))}')
    print(f'[INFO] val unique groups: {len(set(groups_val))}')
    print(f'[INFO] group granularity: {GROUP_GRANULARITY}')
    print('[INFO] held-out validation split is group-aware and uses disjoint subject_id groups.')

    print_label_distribution(y_train_test, split_name=f'train_test {split_name}')
    print_label_distribution(y_val, split_name=f'val {split_name}')

    val_loader = build_validation_loader(X_val, y_val, config)

    return X_train_test, y_train_test, groups_train_test, meta_train_test, val_loader, meta_val


if __name__ == '__main__':
    # 方便单独测试
    from config import Path

    path = Path()
    data_generator(path_labels=path.path_labels, path_dataset=path.path_TF)
