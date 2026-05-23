import os
import numpy as np

from scipy.fftpack import fft
from scipy import signal
from tqdm import tqdm

from args import Path


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


def data_normalize(dataset, channel, save_dir):
    """Normalize datasets of each channel to zero mean and unit variance."""
    print(f'[INFO] Checking inf values before normalization for channel={channel} ...')
    for i in tqdm(range(dataset.shape[0])):
        if np.any(np.isinf(dataset[i])):
            for j in range(29):
                if np.any(np.isinf(dataset[i][j])):
                    for k in range(128):
                        if np.isinf(dataset[i][j][k]):
                            print('location of inf: ', i, ',', j, ',', k)
                            if k == 0:
                                if j == 0:
                                    dataset[i][j][k] = dataset[i][j + 1][k]
                                else:
                                    dataset[i][j][k] = dataset[i][j - 1][k]
                            else:
                                dataset[i][j][k] = dataset[i][j][k - 1]

    mean_val = np.mean(dataset)
    std_val = np.std(dataset)

    print(f'[INFO] Before normalization: mean={mean_val:.6f}, std={std_val:.6f}')
    dataset = (dataset - mean_val) / std_val

    has_inf = np.any(np.isinf(dataset))
    has_nan = np.any(np.isnan(dataset))

    save_path = os.path.join(save_dir, f'TF_{channel}_mean_std.npy')
    if (not has_inf) and (not has_nan):
        np.save(save_path, dataset)
        print(f'[INFO] Saved {save_path}, shape={dataset.shape}')
        print(f'[INFO] After normalization: mean={np.mean(dataset):.6f}, std={np.std(dataset):.6f}')
    else:
        print(f'[ERROR] {channel} still contains inf or nan, not saved.')


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

    # 3. 处理三个通道的 TF 图
    for channel in ['EEG_Fpz-Cz', 'EEG_Pz-Oz', 'EOG']:
        print('\n' + '-' * 15, f'Processing channel: {channel}', '-' * 15)

        data_channel = data_array_concat(path_array=os.path.join(path.path_raw_data, channel))
        X = np.zeros([data_channel.shape[0], 29, int(nfft / 2)], dtype=np.float32)

        print('Transform to TF images:')
        for i in tqdm(range(data_channel.shape[0])):
            Xi = spectrogram(data_channel[i, :], win_size * fs, overlap * fs, nfft)
            Xi = 20 * np.log10(np.abs(Xi) + 1e-8)
            X[i, :, :] = Xi[:, 1:129]

        print(f'[INFO] TF image shape for {channel}: {X.shape}')
        print('Normalize:')
        data_normalize(dataset=X, channel=channel, save_dir=path.path_TF)


'''
import os
import numpy as np

from scipy.fftpack import fft
from scipy import signal
from tqdm import tqdm

from args import Path


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