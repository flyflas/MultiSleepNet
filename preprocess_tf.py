import os
import csv
import re
import numpy as np

from scipy.fftpack import fft
from scipy import signal
from tqdm import tqdm

from config import Path


CHANNELS = ['EEG_Fpz-Cz', 'EEG_Pz-Oz', 'EOG']
METADATA_FILE_NAME = 'tf_per_file_metadata.csv'
PER_FILE_DIR_NAME = 'per_file'
SLEEP_EDF_SUBJECT_RE = re.compile(r'^[A-Za-z]{2}\d{4}')
NORMALIZATION_STRATEGY = (
    'channel-level global mean/std computed across all files, '
    'then applied unchanged to every per-file TF array'
)


def get_npy_file_list(path_array):
    """Get sorted .npy file list."""
    file_list = sorted([f for f in os.listdir(path_array) if f.endswith('.npy')])
    return file_list


def strip_suffix(file_name, channel):
    """Remove channel-specific suffix and keep sample basename."""
    if channel == 'EEG_Fpz-Cz':
        return file_name.replace('_EEG_Fpz-Cz.npy', '')
    elif channel == 'EEG_Pz-Oz':
        return file_name.replace('_EEG_Pz-Oz.npy', '')
    elif channel == 'EOG':
        return file_name.replace('_EOG.npy', '')
    elif channel == 'labels':
        return file_name.replace('_label.npy', '')
    else:
        return file_name


def check_raw_data_alignment(path_raw_data):
    """Check whether EEG/EOG/labels files are aligned before TF transform."""
    channels = ['EEG_Fpz-Cz', 'EEG_Pz-Oz', 'EOG', 'labels']
    file_dict = {}

    print('=' * 80)
    print('[CHECK] Checking raw_data file alignment...')
    for channel in channels:
        channel_path = os.path.join(path_raw_data, channel)
        file_list = get_npy_file_list(channel_path)
        file_dict[channel] = file_list

        print(f'\n[{channel}] num_files = {len(file_list)}')
        for f in file_list[:10]:
            print(f'  {f}')

    lengths = [len(file_dict[ch]) for ch in channels]
    if len(set(lengths)) != 1:
        raise ValueError(f'[ERROR] File counts are inconsistent across channels: {dict(zip(channels, lengths))}')

    ref_keys = [strip_suffix(f, 'EEG_Fpz-Cz') for f in file_dict['EEG_Fpz-Cz']]
    pz_keys = [strip_suffix(f, 'EEG_Pz-Oz') for f in file_dict['EEG_Pz-Oz']]
    eog_keys = [strip_suffix(f, 'EOG') for f in file_dict['EOG']]
    label_keys = [strip_suffix(f, 'labels') for f in file_dict['labels']]

    mismatch_found = False
    for i, (k1, k2, k3, k4) in enumerate(zip(ref_keys, pz_keys, eog_keys, label_keys)):
        if not (k1 == k2 == k3 == k4):
            print(f'[ERROR] Mismatch at index {i}:')
            print(f'  EEG_Fpz-Cz : {file_dict["EEG_Fpz-Cz"][i]}')
            print(f'  EEG_Pz-Oz  : {file_dict["EEG_Pz-Oz"][i]}')
            print(f'  EOG        : {file_dict["EOG"][i]}')
            print(f'  labels     : {file_dict["labels"][i]}')
            mismatch_found = True
            break

    if mismatch_found:
        raise ValueError('[ERROR] Raw data files are NOT aligned. Please fix filenames/order first.')
    else:
        print('\n[OK] Raw data files are aligned across EEG_Fpz-Cz / EEG_Pz-Oz / EOG / labels.')
    print('=' * 80)


def check_label_distribution(path_labels):
    """Check overall label distribution."""
    label_files = get_npy_file_list(path_labels)
    all_labels = []

    print('\n' + '=' * 80)
    print('[CHECK] Checking overall label distribution...')
    for f in tqdm(label_files, desc='Loading label files'):
        y = np.load(os.path.join(path_labels, f))
        all_labels.append(y)

    all_labels = np.concatenate(all_labels, axis=0)
    unique, counts = np.unique(all_labels, return_counts=True)

    print('[INFO] Overall label distribution:')
    total = len(all_labels)
    for u, c in zip(unique, counts):
        print(f'  class {u}: {c} ({c / total:.6f})')

    print(f'[INFO] Total labels: {total}')
    print('=' * 80)


def data_array_concat(path_array):
    """Concat data from each subject."""
    file_list = get_npy_file_list(path_array)
    data_list = []

    print(f'Preparing dataset from: {path_array}')
    for f in tqdm(file_list):
        data = np.load(os.path.join(path_array, f)).astype('float32')
        data_list.append(data)

    data_channel = np.concatenate(data_list, axis=0)
    data_channel = np.squeeze(data_channel, axis=1)

    print(f'[INFO] Concatenated shape from {path_array}: {data_channel.shape}')
    return data_channel


def load_channel_data_with_boundaries(path_array, channel):
    """Load one raw channel and keep file boundaries for per-file TF output."""
    file_list = get_npy_file_list(path_array)
    sample_ids = []
    epoch_counts = []
    data_list = []

    print(f'Preparing dataset from: {path_array}')
    for f in tqdm(file_list):
        data = np.load(os.path.join(path_array, f)).astype('float32')
        sample_id = strip_suffix(f, channel)

        if data.ndim == 3 and data.shape[1] == 1:
            data = np.squeeze(data, axis=1)
        elif data.ndim != 2:
            raise ValueError(
                f'[ERROR] Expected {f} to have shape [epochs, 1, samples] or [epochs, samples], '
                f'got {data.shape}'
            )

        sample_ids.append(sample_id)
        epoch_counts.append(data.shape[0])
        data_list.append(data)

    if not data_list:
        raise ValueError(f'[ERROR] No .npy files found in {path_array}')

    data_channel = np.concatenate(data_list, axis=0)
    print(f'[INFO] Concatenated shape from {path_array}: {data_channel.shape}')
    return data_channel, sample_ids, epoch_counts


def infer_subject_id(sample_id):
    """Infer Sleep-EDF subject id using the same 6-char key as prepare_dataset.py."""
    match = SLEEP_EDF_SUBJECT_RE.match(sample_id)
    if match:
        return match.group(0)
    return sample_id[:6] if len(sample_id) >= 6 else sample_id


def spectrogram(x, window, n_overlap, nfft):
    """
    Transform to time-frequency images.
    This function imitates Matlab spectrogram.
    """
    len_x = len(x)
    step = window - n_overlap
    nn = nfft // 2 + 1
    num_win = int(np.floor((len_x - n_overlap) / step))
    spectrogram_data = []

    win = signal.windows.hamming(window)
    for i in range(num_win):
        subdata = x[i * step: i * step + window]
        F = fft(subdata * win, n=nfft)
        spectrogram_data.append(F[:nn])

    spectrogram_data = np.array(spectrogram_data)
    return spectrogram_data


def ensure_finite_tf_values(dataset, channel, stage):
    """Replace non-finite TF values before stats so outputs never contain inf/nan."""
    non_finite_mask = ~np.isfinite(dataset)
    non_finite_count = int(np.count_nonzero(non_finite_mask))
    if non_finite_count == 0:
        return dataset

    finite_values = dataset[~non_finite_mask]
    if finite_values.size == 0:
        raise ValueError(f'[ERROR] {channel} has no finite TF values during {stage}.')

    replacement = float(np.mean(finite_values, dtype=np.float64))
    dataset = dataset.copy()
    dataset[non_finite_mask] = replacement
    print(
        f'[WARN] Replaced {non_finite_count} non-finite TF values for channel={channel} '
        f'during {stage} with finite channel mean={replacement:.6f}.'
    )
    return dataset


def save_channel_global_stats(dataset, channel, save_dir):
    """Compute and save global channel-level normalization stats."""
    mean_val = float(np.mean(dataset, dtype=np.float64))
    std_val = float(np.std(dataset, dtype=np.float64))

    if (not np.isfinite(mean_val)) or (not np.isfinite(std_val)) or std_val <= 0.0:
        raise ValueError(
            f'[ERROR] Invalid normalization stats for {channel}: '
            f'mean={mean_val}, std={std_val}'
        )

    stats = np.array([mean_val, std_val], dtype=np.float64)
    stats_path = os.path.join(save_dir, f'TF_{channel}_mean_std_stats.npy')
    np.save(stats_path, stats)

    print(f'[INFO] Normalization strategy for {channel}: {NORMALIZATION_STRATEGY}.')
    print(f'[INFO] Stats source for {channel}: computed from concatenated TF features in this run.')
    print(f'[INFO] Saved channel global stats: {stats_path}')
    print(f'[INFO] Stats values for {channel}: mean={mean_val:.6f}, std={std_val:.6f}')
    return mean_val, std_val, stats_path


def normalize_with_channel_global_stats(dataset, channel, mean_val, std_val, stats_path, save_dir):
    """Normalize one channel using precomputed global stats and save legacy output."""
    print(f'[INFO] Applying stats source for {channel}: {stats_path}')
    normalized = apply_channel_global_stats(
        dataset=dataset,
        channel=channel,
        mean_val=mean_val,
        std_val=std_val,
        stats_path=stats_path,
        stage='legacy global normalized TF array'
    )

    save_path = os.path.join(save_dir, f'TF_{channel}_mean_std.npy')
    np.save(save_path, normalized)
    print(f'[INFO] Saved legacy global normalized TF array: {save_path}, shape={normalized.shape}')
    print(f'[INFO] After normalization: mean={np.mean(normalized):.6f}, std={np.std(normalized):.6f}')
    return normalized


def apply_channel_global_stats(dataset, channel, mean_val, std_val, stats_path, stage):
    """Apply one channel-level global mean/std to a TF array."""
    normalized = (
        (dataset.astype(np.float32, copy=False) - np.float32(mean_val)) / np.float32(std_val)
    ).astype(np.float32, copy=False)
    normalized = ensure_finite_tf_values(
        normalized,
        channel=channel,
        stage=f'after channel-level global normalization for {stage}'
    )

    if not np.all(np.isfinite(normalized)):
        raise ValueError(
            f'[ERROR] {channel} contains inf or nan after applying stats source {stats_path} '
            f'for {stage}.'
        )
    return normalized


def data_normalize(dataset, channel, save_dir, return_stats=False):
    """Normalize a channel with one global mean/std shared by all files."""
    print(f'[INFO] Normalization strategy: {NORMALIZATION_STRATEGY}.')
    print('[INFO] This path does not perform per-file normalization.')
    dataset = ensure_finite_tf_values(
        dataset=dataset,
        channel=channel,
        stage='before channel-level global stats'
    )
    mean_val, std_val, stats_path = save_channel_global_stats(
        dataset=dataset,
        channel=channel,
        save_dir=save_dir
    )
    normalized = normalize_with_channel_global_stats(
        dataset=dataset,
        channel=channel,
        mean_val=mean_val,
        std_val=std_val,
        stats_path=stats_path,
        save_dir=save_dir
    )
    if return_stats:
        return normalized, mean_val, std_val, stats_path, dataset
    return normalized


def transform_to_tf(data_channel, fs, overlap, nfft, win_size):
    """Convert one concatenated raw channel to TF images."""
    X = np.zeros([data_channel.shape[0], 29, int(nfft / 2)], dtype=np.float32)

    print('Transform to TF images:')
    for i in tqdm(range(data_channel.shape[0])):
        Xi = spectrogram(data_channel[i, :], win_size * fs, overlap * fs, nfft)
        Xi = 20 * np.log10(np.abs(Xi) + 1e-8)
        X[i, :, :] = Xi[:, 1:129]

    return X


def save_per_file_tf(dataset, channel, sample_ids, epoch_counts, save_dir, mean_val, std_val, stats_path):
    """Save globally normalized per-file TF arrays and return metadata paths."""
    output_dir = os.path.join(save_dir, PER_FILE_DIR_NAME, channel)
    os.makedirs(output_dir, exist_ok=True)

    if int(np.sum(epoch_counts)) != dataset.shape[0]:
        raise ValueError(
            f'[ERROR] {channel} epoch boundary total ({np.sum(epoch_counts)}) '
            f'does not match dataset size ({dataset.shape[0]})'
        )

    print(f'[INFO] Per-file TF arrays for {channel} will use stats source: {stats_path}')
    paths = {}
    offset = 0
    for sample_id, num_epochs in zip(sample_ids, epoch_counts):
        end = offset + num_epochs
        sample_tf = apply_channel_global_stats(
            dataset=dataset[offset:end],
            channel=channel,
            mean_val=mean_val,
            std_val=std_val,
            stats_path=stats_path,
            stage=f'per-file TF array sample_id={sample_id}'
        )
        if sample_tf.shape != (num_epochs, 29, 128):
            raise ValueError(
                f'[ERROR] {sample_id} {channel} TF shape mismatch: '
                f'expected ({num_epochs}, 29, 128), got {sample_tf.shape}'
            )
        if not np.all(np.isfinite(sample_tf)):
            raise ValueError(
                f'[ERROR] {sample_id} {channel} contains inf or nan after '
                'channel-level global normalization.'
            )

        save_path = os.path.join(output_dir, f'{sample_id}_TF_{channel}_mean_std.npy')
        np.save(save_path, sample_tf)
        paths[sample_id] = os.path.relpath(save_path, save_dir)
        offset = end

    print(f'[INFO] Saved {len(paths)} per-file TF arrays for {channel} under {output_dir}')
    return paths


def save_per_file_labels(path_labels, save_dir, sample_ids, epoch_counts):
    """Copy aligned labels into the per-file TF output tree."""
    label_files = get_npy_file_list(path_labels)
    label_sample_ids = [strip_suffix(f, 'labels') for f in label_files]
    if label_sample_ids != sample_ids:
        raise ValueError(
            '[ERROR] Label sample order does not match channel sample order: '
            f'labels={label_sample_ids[:5]}, channels={sample_ids[:5]}'
        )

    output_dir = os.path.join(save_dir, PER_FILE_DIR_NAME, 'labels')
    os.makedirs(output_dir, exist_ok=True)

    paths = {}
    for f, sample_id, num_epochs in zip(label_files, sample_ids, epoch_counts):
        labels = np.load(os.path.join(path_labels, f)).astype(np.int64)
        if labels.shape != (num_epochs,):
            raise ValueError(
                f'[ERROR] {sample_id} label shape mismatch: '
                f'expected ({num_epochs},), got {labels.shape}'
            )

        save_path = os.path.join(output_dir, f'{sample_id}_label.npy')
        np.save(save_path, labels)
        paths[sample_id] = os.path.relpath(save_path, save_dir)

    print(f'[INFO] Saved {len(paths)} per-file label arrays under {output_dir}')
    return paths


def write_metadata(save_dir, sample_ids, epoch_counts, channel_paths, label_paths):
    """Write per-file TF manifest for future context-window construction."""
    metadata_path = os.path.join(save_dir, METADATA_FILE_NAME)
    fieldnames = [
        'sample_id',
        'subject_id',
        'num_epochs',
        'EEG Fpz-Cz TF path',
        'EEG Pz-Oz TF path',
        'EOG TF path',
        'label path',
    ]

    with open(metadata_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sample_id, num_epochs in zip(sample_ids, epoch_counts):
            writer.writerow({
                'sample_id': sample_id,
                'subject_id': infer_subject_id(sample_id),
                'num_epochs': int(num_epochs),
                'EEG Fpz-Cz TF path': channel_paths['EEG_Fpz-Cz'][sample_id],
                'EEG Pz-Oz TF path': channel_paths['EEG_Pz-Oz'][sample_id],
                'EOG TF path': channel_paths['EOG'][sample_id],
                'label path': label_paths[sample_id],
            })

    print(f'[INFO] Saved per-file TF metadata: {metadata_path}')
    return metadata_path


if __name__ == '__main__':
    path = Path()

    fs = 100
    overlap = 1
    nfft = 256
    win_size = 2

    # 1. 先检查原始 raw_data 的四个目录是否严格对齐
    check_raw_data_alignment(path.path_raw_data)

    # 2. 检查 labels 总体分布
    check_label_distribution(os.path.join(path.path_raw_data, 'labels'))

    os.makedirs(path.path_TF, exist_ok=True)

    # 3. 处理三个通道的 TF 图；保留旧全局输出，同时保存 per-file 输出和 metadata。
    reference_sample_ids = None
    reference_epoch_counts = None
    channel_paths = {}

    for channel in CHANNELS:
        print('\n' + '-' * 15, f'Processing channel: {channel}', '-' * 15)

        data_channel, sample_ids, epoch_counts = load_channel_data_with_boundaries(
            path_array=os.path.join(path.path_raw_data, channel),
            channel=channel
        )

        if reference_sample_ids is None:
            reference_sample_ids = sample_ids
            reference_epoch_counts = epoch_counts
        elif sample_ids != reference_sample_ids or epoch_counts != reference_epoch_counts:
            raise ValueError(f'[ERROR] {channel} file boundaries do not match the reference channel.')

        X = transform_to_tf(data_channel, fs=fs, overlap=overlap, nfft=nfft, win_size=win_size)
        print(f'[INFO] TF image shape for {channel}: {X.shape}')
        print('Normalize:')
        _, mean_val, std_val, stats_path, finite_X = data_normalize(
            dataset=X,
            channel=channel,
            save_dir=path.path_TF,
            return_stats=True
        )
        channel_paths[channel] = save_per_file_tf(
            dataset=finite_X,
            channel=channel,
            sample_ids=sample_ids,
            epoch_counts=epoch_counts,
            save_dir=path.path_TF,
            mean_val=mean_val,
            std_val=std_val,
            stats_path=stats_path
        )

    label_paths = save_per_file_labels(
        path_labels=os.path.join(path.path_raw_data, 'labels'),
        save_dir=path.path_TF,
        sample_ids=reference_sample_ids,
        epoch_counts=reference_epoch_counts
    )
    write_metadata(
        save_dir=path.path_TF,
        sample_ids=reference_sample_ids,
        epoch_counts=reference_epoch_counts,
        channel_paths=channel_paths,
        label_paths=label_paths
    )


'''
import os
import numpy as np

from scipy.fftpack import fft
from scipy import signal
from tqdm import tqdm

from config import Path


def data_array_concat(path_array):
    """concat data from each subject"""
    dir_PSG = os.listdir(path_array)
    first = True
    print('Preparing dataset:')
    for f in tqdm(dir_PSG):
        if first:
            data_channel = np.load(os.path.join(path_array, f)).astype('float32')
            first = False
        else:
            temp = np.load(os.path.join(path_array, f)).astype('float32')
            data_channel = np.append(data_channel, temp, axis=0)
    data_channel = np.squeeze(data_channel, axis=1)
    return data_channel


def spectrogram(x, window, n_overlap, nfft):
    """
    Transform to time-frequency images. This function imitates function spectrogram in Matlab
    Args:
        x (numpy array): Data
        window (int): Size of window function
        n_overlap (int):Number of coincidence points between two segments
        nfft (int): Number of points during Fast Fourier Transform
    """
    len_x = len(x)
    step = window - n_overlap
    nn = nfft // 2 + 1
    num_win = int(np.floor((len_x - n_overlap) / (window - n_overlap)))
    spectrogram_data = []
    # Hamming window default
    win = signal.hamming(window)
    for i in range(num_win):
        subdata = x[i * step: i * step + window]
        F = fft(subdata * win, n=nfft)
        spectrogram_data.append(F[:nn])
    spectrogram_data = np.array(spectrogram_data)
    return spectrogram_data


def data_normalize(dataset, channel):
    """normalize datasets of each channel to zero mean and unit variance"""
    for i in tqdm(range(dataset.shape[0])):
        if True in np.isinf(dataset[i]):
            for j in range(29):
                if True in np.isinf(dataset[i][j]):
                    for k in range(128):
                        if np.isinf(dataset[i][j][k]):
                            if k != 127:
                                print('location of inf: ', i, ',', j, ',', k)
                            if k == 0:
                                if j == 0:
                                    dataset[i][j][k] = dataset[i][j+1][k]
                                else:
                                    dataset[i][j][k] = dataset[i][j-1][k]
                            else:
                                dataset[i][j][k] = dataset[i][j][k-1]

    dataset = (dataset - np.mean(dataset)) / np.std(dataset)

    ans1 = np.isinf(dataset)
    ans2 = np.isnan(dataset)

    if not ((True in ans1) and (True in ans2)):
        np.save('./data/sleepEDF-78/data_array/TF_data/TF_{}_mean_std.npy'.format(channel), dataset)


if __name__ == '__main__':
    path = Path()

    fs = 100
    overlap = 1
    nfft = 256
    win_size = 2


    for channel in ['EEG_Fpz-Cz', 'EEG_Pz-Oz', 'EOG', 'labels']:
        file_list = sorted([f for f in os.listdir(os.path.join(path.path_raw_data, channel)) if f.endswith('.npy')])
        print(channel, len(file_list))
        print(file_list[:10])
        print('-' * 50)

    for channel in ['EEG_Fpz-Cz', 'EEG_Pz-Oz', 'EOG']:
        print('-' * 15, 'Processing channel:{}'.format(channel), '-' * 15)
        data_channel = data_array_concat(path_array=os.path.join(path.path_raw_data, channel))
        X = np.zeros([data_channel.shape[0], 29, int(nfft / 2)])
        print('Transform to TF images:')
        for i in tqdm(range(data_channel.shape[0])):
            Xi = spectrogram(data_channel[i, :], win_size * fs, overlap * fs, nfft)
            Xi = 20 * np.log10(abs(Xi))
            X[i, :, :] = Xi[:, 1:129]

        print('Normalize:')
        data_normalize(dataset=X, channel=channel)
'''
