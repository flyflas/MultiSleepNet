import os
import shutil

import mne
import numpy as np

from config import Path


def prepare_SleepEDF_20(path_PSG, path_hypnogram, save_path):
    """Extract 30-s epochs from EDF files."""

    # 1) 把误放在 PSG 目录下的 Hypnogram 文件移动到单独目录
    for file in os.listdir(path_PSG):
        if file.endswith(".edf") and "Hypnogram" in file:
            original_path = os.path.join(path_PSG, file)
            target_path = os.path.join(path_hypnogram, file)

            # 避免重复运行时报错
            if not os.path.exists(target_path):
                shutil.move(original_path, target_path)

    annotation_desc_2_event_id = {
        'Sleep stage W': 1,
        'Sleep stage 1': 2,
        'Sleep stage 2': 3,
        'Sleep stage 3': 4,
        'Sleep stage 4': 4,
        'Sleep stage R': 5
    }

    event_id = {
        'Sleep stage W': 1,
        'Sleep stage 1': 2,
        'Sleep stage 2': 3,
        'Sleep stage 3/4': 4,
        'Sleep stage R': 5
    }

    event_id_with_no_N3N4 = {
        'Sleep stage W': 1,
        'Sleep stage 1': 2,
        'Sleep stage 2': 3,
        'Sleep stage R': 5
    }

    # 2) 只保留真正的 PSG / Hypnogram EDF 文件
    dir_PSG = sorted([
        f for f in os.listdir(path_PSG)
        if f.endswith('.edf') and 'PSG' in f
    ])

    dir_annotation = sorted([
        f for f in os.listdir(path_hypnogram)
        if f.endswith('.edf') and 'Hypnogram' in f
    ])

    # 3) 按受试者编号建立 Hypnogram 映射，而不是直接 zip
    #    Sleep-EDF 文件名前 6 位如 SC4001 / SC4762 可作为匹配键
    annotation_map = {}
    for ann_file in dir_annotation:
        subject_id = ann_file[:6]
        annotation_map[subject_id] = ann_file

    # 4) 创建保存目录
    for channel in ['EEG_Fpz-Cz', 'EEG_Pz-Oz', 'EOG', 'labels']:
        os.makedirs(os.path.join(save_path, channel), exist_ok=True)

    # 5) 逐个 PSG 文件寻找对应标注
    for psg_file_name in dir_PSG:
        subject_id = psg_file_name[:6]

        if subject_id not in annotation_map:
            print(f'[WARNING] No matched hypnogram for {psg_file_name}, skipped.')
            continue

        ann_file_name = annotation_map[subject_id]
        print('current file: ', psg_file_name, ann_file_name)

        PSG_file = os.path.join(path_PSG, psg_file_name)
        annotation_file = os.path.join(path_hypnogram, ann_file_name)

        try:
            raw_train = mne.io.read_raw_edf(
                PSG_file,
                stim_channel='marker',
                misc=['rectal'],
                preload=True
            )
            annotation_train = mne.read_annotations(annotation_file)
            raw_train.set_annotations(annotation_train, emit_warning=False)

            # 去除前后 30 分钟无关部分
            annotation_train.crop(
                annotation_train[1]['onset'] - 30 * 60,
                annotation_train[-2]['onset'] + 30 * 60
            )
            raw_train.set_annotations(annotation_train, emit_warning=False)

            events_train, sleep_stage_exist = mne.events_from_annotations(
                raw_train,
                event_id=annotation_desc_2_event_id,
                chunk_duration=30.
            )

            tmax = 30. - 1. / raw_train.info['sfreq']

            if len(sleep_stage_exist) <= 4:
                epochs_train = mne.Epochs(
                    raw=raw_train,
                    events=events_train,
                    event_id=event_id_with_no_N3N4,
                    tmin=0.,
                    tmax=tmax,
                    baseline=None,
                    preload=True
                )
            else:
                epochs_train = mne.Epochs(
                    raw=raw_train,
                    events=events_train,
                    event_id=event_id,
                    tmin=0.,
                    tmax=tmax,
                    baseline=None,
                    preload=True
                )

            # 6) 提取通道
            X_train_eeg_FpzCz = epochs_train.copy().pick_channels(['EEG Fpz-Cz']).get_data()
            X_train_eeg_PzOz = epochs_train.copy().pick_channels(['EEG Pz-Oz']).get_data()
            X_train_eog = epochs_train.copy().pick_channels(['EOG horizontal']).get_data()

            y_train = epochs_train.copy().pick_channels(['EEG Fpz-Cz']).events[:, 2]
            y_train = y_train - 1

            # 7) 保存文件，使用 PSG 文件名前 12 位作为样本名
            sample_name = psg_file_name[:12]

            print('current file: ', psg_file_name, ann_file_name)
            print('label unique/count:', np.unique(y_train, return_counts=True))
            np.save(os.path.join(save_path, 'EEG_Fpz-Cz', f'{sample_name}_EEG_Fpz-Cz.npy'), X_train_eeg_FpzCz)
            np.save(os.path.join(save_path, 'EEG_Pz-Oz', f'{sample_name}_EEG_Pz-Oz.npy'), X_train_eeg_PzOz)
            np.save(os.path.join(save_path, 'EOG', f'{sample_name}_EOG.npy'), X_train_eog)
            np.save(os.path.join(save_path, 'labels', f'{sample_name}_label.npy'), y_train)

        except Exception as e:
            print(f'[ERROR] Failed on {psg_file_name} and {ann_file_name}: {e}')
            continue


if __name__ == '__main__':
    path = Path()
    prepare_SleepEDF_20(
        path_PSG=path.path_PSG,
        path_hypnogram=path.path_hypnogram,
        save_path=path.path_raw_data
    )
