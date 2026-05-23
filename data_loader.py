import os
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from config import Config


CHANNELS = ['EEG_Fpz-Cz', 'EEG_Pz-Oz', 'EOG']
CHANNEL_SUFFIXES = {
    'EEG_Fpz-Cz': '_EEG_Fpz-Cz.npy',
    'EEG_Pz-Oz': '_EEG_Pz-Oz.npy',
    'EOG': '_EOG.npy',
    'labels': '_label.npy',
}


def get_npy_file_list(path_array):
    return sorted([f for f in os.listdir(path_array) if f.endswith('.npy')])


def strip_suffix(file_name, channel):
    suffix = CHANNEL_SUFFIXES[channel]
    if not file_name.endswith(suffix):
        raise ValueError(
            f'[ERROR] {channel} file has unexpected suffix: {file_name}. '
            f'Expected suffix: {suffix}'
        )
    return file_name[:-len(suffix)]


def build_channel_file_name(sample_key, channel):
    return f'{sample_key}{CHANNEL_SUFFIXES[channel]}'


def npy_epoch_count(file_path):
    data = np.load(file_path, mmap_mode='r')
    if data.ndim == 0:
        raise ValueError(f'[ERROR] Expected epoch array in {file_path}, got scalar shape.')
    return int(data.shape[0])


def validate_sse_raw_alignment(path_raw_data, label_files):
    """Validate raw SSE channel files against the label/TF concatenation order."""
    if len(label_files) == 0:
        raise ValueError('[ERROR] No label .npy files found.')

    label_keys = [strip_suffix(f, 'labels') for f in label_files]
    if len(set(label_keys)) != len(label_keys):
        raise ValueError('[ERROR] Duplicate label basenames found; cannot build aligned dataset.')
    label_counts = {
        key: npy_epoch_count(os.path.join(path_raw_data, 'labels', label_file))
        for key, label_file in zip(label_keys, label_files)
    }

    print('=' * 80)
    print('[CHECK] Checking SSE raw file alignment against labels/TF order...')
    print(f'[labels] num_files = {len(label_files)}')
    for f in label_files[:10]:
        print(f'  {f}')

    aligned_files = {}
    for channel in CHANNELS:
        channel_path = os.path.join(path_raw_data, channel)
        channel_files = get_npy_file_list(channel_path)
        channel_keys = [strip_suffix(f, channel) for f in channel_files]

        print(f'\n[{channel}] num_files = {len(channel_files)}')
        for f in channel_files[:10]:
            print(f'  {f}')

        if len(channel_files) != len(label_files):
            raise ValueError(
                f'[ERROR] File count mismatch for {channel}: '
                f'{len(channel_files)} raw files != {len(label_files)} label files'
            )

        if len(set(channel_keys)) != len(channel_keys):
            raise ValueError(f'[ERROR] Duplicate {channel} basenames found; cannot build aligned dataset.')

        if channel_keys != label_keys:
            mismatch_index = next(
                (i for i, (raw_key, label_key) in enumerate(zip(channel_keys, label_keys)) if raw_key != label_key),
                None
            )
            missing_from_channel = sorted(set(label_keys) - set(channel_keys))[:5]
            extra_in_channel = sorted(set(channel_keys) - set(label_keys))[:5]
            detail = ''
            if mismatch_index is not None:
                detail = (
                    f' First mismatch at index {mismatch_index}: '
                    f'{channel}={channel_files[mismatch_index]}, labels={label_files[mismatch_index]}.'
                )
            raise ValueError(
                f'[ERROR] {channel} raw files are not basename-aligned with labels/TF order.'
                f'{detail} Missing from {channel}: {missing_from_channel}; extra in {channel}: {extra_in_channel}'
            )

        aligned_channel_files = [build_channel_file_name(key, channel) for key in label_keys]
        for key, channel_file in zip(label_keys, aligned_channel_files):
            label_count = label_counts[key]
            raw_count = npy_epoch_count(os.path.join(channel_path, channel_file))
            if raw_count != label_count:
                raise ValueError(
                    f'[ERROR] Per-file epoch count mismatch for {key}: '
                    f'{channel_file} has {raw_count} raw epochs, '
                    f'{build_channel_file_name(key, "labels")} has {label_count} labels'
                )

        aligned_files[channel] = aligned_channel_files

    print('\n[OK] SSE raw files are basename-aligned and per-file epoch-count aligned with labels/TF order.')
    print('=' * 80)
    return aligned_files


def print_label_distribution(labels, split_name='labels'):
    """Print label distribution."""
    if isinstance(labels, torch.Tensor):
        labels_np = labels.cpu().numpy()
    else:
        labels_np = labels

    unique, counts = np.unique(labels_np, return_counts=True)
    total = len(labels_np)

    print(f'[{split_name}] label distribution:')
    for u, c in zip(unique, counts):
        print(f'  class {u}: {c} ({c / total:.6f})')


def data_array_concat(path_array, file_list=None):
    if file_list is None:
        file_list = get_npy_file_list(path_array)
    data_list = []
    for f in file_list:
        data = np.load(os.path.join(path_array, f)).astype('float32')
        data_list.append(data)

    data_channel = np.concatenate(data_list, axis=0)
    return np.squeeze(data_channel, axis=1)


def build_sse_windows(data_channel, config):
    stride = config.sse_window_size // 2
    num_samples = data_channel.shape[0]
    sse_data = np.empty(
        (num_samples, config.sse_num_windows, config.sse_window_size),
        dtype=np.float32
    )

    for i in range(config.sse_num_windows):
        start = i * stride
        end = start + config.sse_window_size
        sse_data[:, i, :] = data_channel[:, start:end]

    return sse_data


def load_sse_dataset(path_raw_data, config, label_files):
    aligned_files = validate_sse_raw_alignment(path_raw_data, label_files)
    channel_data = []
    for channel in CHANNELS:
        data_channel = data_array_concat(
            os.path.join(path_raw_data, channel),
            file_list=aligned_files[channel]
        )
        required_len = (config.sse_num_windows - 1) * (config.sse_window_size // 2) + config.sse_window_size
        if data_channel.shape[1] < required_len:
            raise ValueError(
                f'[ERROR] {channel} raw epoch length ({data_channel.shape[1]}) < required SSE length ({required_len})'
            )
        channel_data.append(build_sse_windows(data_channel, config))

    dataset = np.stack(channel_data, axis=1)
    mean_val = np.mean(dataset, axis=(0, 2, 3), keepdims=True)
    std_val = np.std(dataset, axis=(0, 2, 3), keepdims=True)
    std_val = np.where(std_val == 0, 1.0, std_val)
    dataset = (dataset - mean_val) / std_val

    return torch.from_numpy(dataset.astype('float32'))


def data_generator(path_labels, path_dataset):
    config = Config()

    # 1. 一定要排序，保证和 preprocess_tf.py 的拼接顺序一致
    label_files = get_npy_file_list(path_labels)

    print('[INFO] label files (first 10):')
    for f in label_files[:10]:
        print(f'  {f}')

    # 2. 按排序后的文件顺序拼接 labels
    label_list = []
    for f in label_files:
        y = np.load(os.path.join(path_labels, f))
        label_list.append(y)

    labels = np.concatenate(label_list, axis=0).astype(np.int64)
    labels = torch.from_numpy(labels)

    # 3. 读取三个通道的 TF 数据
    dataset_EEG_FpzCz = np.load(
        os.path.join(path_dataset, 'TF_EEG_Fpz-Cz_mean_std.npy')
    ).astype('float32')

    dataset_EEG_PzOz = np.load(
        os.path.join(path_dataset, 'TF_EEG_Pz-Oz_mean_std.npy')
    ).astype('float32')

    dataset_EOG = np.load(
        os.path.join(path_dataset, 'TF_EOG_mean_std.npy')
    ).astype('float32')

    # 4. 堆叠成 [N, 3, 29, 128]
    tf_dataset = np.stack((dataset_EEG_FpzCz, dataset_EEG_PzOz, dataset_EOG), axis=1)
    tf_dataset = torch.from_numpy(tf_dataset)

    raw_data_root = os.path.dirname(path_labels)
    sse_dataset = load_sse_dataset(raw_data_root, config, label_files)

    print(f'[INFO] tf dataset shape: {tf_dataset.shape}')
    print(f'[INFO] sse dataset shape: {sse_dataset.shape}')
    print(f'[INFO] labels shape: {labels.shape}')

    # 5. 安全检查：样本数必须一致
    if len(tf_dataset) != len(labels):
        raise ValueError(
            f'[ERROR] TF dataset size ({len(tf_dataset)}) != labels size ({len(labels)})'
        )
    if len(sse_dataset) != len(labels):
        raise ValueError(
            f'[ERROR] SSE dataset size ({len(sse_dataset)}) != labels size ({len(labels)})'
        )

    print_label_distribution(labels, split_name='full dataset')

    # 6. 固定留出验证集
    tf_train_test, tf_val, sse_train_test, sse_val, y_train_test, y_val = train_test_split(
        tf_dataset,
        sse_dataset,
        labels,
        # This test_size is the held-out validation ratio, not the K-fold test ratio.
        test_size=config.val_ratio,
        random_state=0,
        stratify=labels
    )

    print(f'[INFO] train_test size: {len(tf_train_test)}')
    print(f'[INFO] val size: {len(tf_val)}')

    print_label_distribution(y_train_test, split_name='train_test')
    print_label_distribution(y_val, split_name='val')

    val_set = TensorDataset(tf_val, sse_val, y_val)
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=12,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    return tf_train_test, sse_train_test, y_train_test, val_loader


if __name__ == '__main__':
    # 方便单独测试
    from config import Path

    path = Path()
    data_generator(path_labels=path.path_labels, path_dataset=path.path_TF)



'''
import os
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from config import Config, Path


def data_generator(path_labels, path_dataset):
    config = Config()
    dir_annotation = os.listdir(path_labels)

    first = True
    for f in dir_annotation:
        if first:
            labels = np.load(os.path.join(path_labels, f))
            first = False
        else:
            temp = np.load(os.path.join(path_labels, f))
            labels = np.append(labels, temp, axis=0)
    labels = torch.from_numpy(labels)

    dataset_EEG_FpzCz = np.load(os.path.join(path_dataset, 'TF_EEG_Fpz-Cz_mean_std.npy')).astype('float32')
    dataset_EEG_PzOz = np.load(os.path.join(path_dataset, 'TF_EEG_Pz-Oz_mean_std.npy')).astype('float32')
    dataset_EOG = np.load(os.path.join(path_dataset, 'TF_EOG_mean_std.npy')).astype('float32')

    dataset = np.stack((dataset_EEG_FpzCz, dataset_EEG_PzOz, dataset_EOG), axis=1)
    dataset = torch.from_numpy(dataset)

    print('dataset: ', dataset.shape)

    # hold out the validation set
    X_train_test, X_val, y_train_test, y_val = train_test_split(dataset, labels, test_size=1/(config.num_fold+1), random_state=0, stratify=labels)

    val_set = TensorDataset(X_val, y_val)
    #val_loader = DataLoader(dataset=val_set, batch_size=config.batch_size, shuffle=False)
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    print('val_set:', len(X_val))
    return X_train_test, y_train_test, val_loader
'''
