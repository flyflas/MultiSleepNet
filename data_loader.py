import os
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from args import Config


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


def data_generator(path_labels, path_dataset):
    config = Config()

    # 1. 一定要排序，保证和 data_preprocess_TF.py 的拼接顺序一致
    label_files = sorted([f for f in os.listdir(path_labels) if f.endswith('.npy')])

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
    dataset = np.stack((dataset_EEG_FpzCz, dataset_EEG_PzOz, dataset_EOG), axis=1)
    dataset = torch.from_numpy(dataset)

    print(f'[INFO] dataset shape: {dataset.shape}')
    print(f'[INFO] labels shape: {labels.shape}')

    # 5. 安全检查：样本数必须一致
    if len(dataset) != len(labels):
        raise ValueError(
            f'[ERROR] dataset size ({len(dataset)}) != labels size ({len(labels)})'
        )

    print_label_distribution(labels, split_name='full dataset')

    # 6. 固定留出验证集
    X_train_test, X_val, y_train_test, y_val = train_test_split(
        dataset,
        labels,
        test_size=1 / (config.num_fold + 1),
        random_state=0,
        stratify=labels
    )

    print(f'[INFO] train_test size: {len(X_train_test)}')
    print(f'[INFO] val size: {len(X_val)}')

    print_label_distribution(y_train_test, split_name='train_test')
    print_label_distribution(y_val, split_name='val')

    val_set = TensorDataset(X_val, y_val)
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=12,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    return X_train_test, y_train_test, val_loader


if __name__ == '__main__':
    # 方便单独测试
    from args import Path

    path = Path()
    data_generator(path_labels=path.path_labels, path_dataset=path.path_TF)



'''
import os
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from args import Config, Path


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