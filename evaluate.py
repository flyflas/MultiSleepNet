import os
import numpy as np
from tqdm import tqdm

from sklearn.metrics import (
    recall_score,
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    balanced_accuracy_score,
)

import torch
from torch.utils.data import TensorDataset, DataLoader

from model import Transformer
from data_loader import data_generator
from config import Config, Path
from mlflow_utils import MLflowTracker


CLASS_NAMES = ['Wake', 'N1', 'N2', 'N3', 'REM']
SPLIT_METADATA_FILE = 'split_metadata.npz'
SPLIT_RANDOM_STATE = 0
SPLIT_SHUFFLE = True
GROUP_GRANULARITY = 'subject_id'
CONTEXT_SIZE = 5
LEFT_CONTEXT = 2
RIGHT_CONTEXT = 2


def specificity(y_true, y_pred, n=5):
    spec = []
    con_mat = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    for i in range(n):
        number = np.sum(con_mat[:, :])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i, :]) - tp
        fp = np.sum(con_mat[:, i]) - tp
        tn = number - tp - fn - fp
        spec1 = tn / (tn + fp + 1e-12)
        spec.append(spec1)
    return np.mean(spec)


def class_wise_specificity(con_mat):
    spec_list = []
    total = np.sum(con_mat)

    for i in range(con_mat.shape[0]):
        tp = con_mat[i, i]
        fn = np.sum(con_mat[i, :]) - tp
        fp = np.sum(con_mat[:, i]) - tp
        tn = total - tp - fn - fp
        spec = tn / (tn + fp + 1e-12)
        spec_list.append(spec)

    return np.array(spec_list, dtype=np.float64)


def class_wise_evaluate(con_mat):
    """
    columns: precision, recall, f1, specificity, support
    """
    num_classes = con_mat.shape[0]
    class_wise_mat = np.empty((num_classes, 5), dtype=np.float64)

    spec_list = class_wise_specificity(con_mat)

    for i in range(num_classes):
        precision = con_mat[i, i] / (np.sum(con_mat[:, i]) + 1e-12)
        recall = con_mat[i, i] / (np.sum(con_mat[i, :]) + 1e-12)
        f1 = (2 * precision * recall) / (precision + recall + 1e-12)
        support = np.sum(con_mat[i, :])

        class_wise_mat[i, 0] = precision
        class_wise_mat[i, 1] = recall
        class_wise_mat[i, 2] = f1
        class_wise_mat[i, 3] = spec_list[i]
        class_wise_mat[i, 4] = support

    return class_wise_mat


def print_class_wise_result(class_wise_result, class_names=CLASS_NAMES):
    print('\n===== CLASS-WISE METRICS =====')
    print(f'{"Class":<8} {"Precision":>10} {"Recall":>10} {"F1":>10} {"Spec":>10} {"Support":>10}')
    for i, name in enumerate(class_names):
        precision, recall, f1, spec, support = class_wise_result[i]
        print(f'{name:<8} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {spec:>10.4f} {int(support):>10}')


def save_evaluation_artifacts(confusion_mat, class_wise_result, output_dir='./Kfold_models/evaluation'):
    os.makedirs(output_dir, exist_ok=True)

    confusion_path = os.path.join(output_dir, 'confusion_matrix.npy')
    class_wise_path = os.path.join(output_dir, 'class_wise_metrics.npy')
    class_wise_csv_path = os.path.join(output_dir, 'class_wise_metrics.csv')

    np.save(confusion_path, confusion_mat)
    np.save(class_wise_path, class_wise_result)

    header = 'class,precision,recall,f1,specificity,support'
    rows = []
    for i, class_name in enumerate(CLASS_NAMES):
        precision, recall, f1, spec, support = class_wise_result[i]
        rows.append(f'{class_name},{precision},{recall},{f1},{spec},{int(support)}')
    with open(class_wise_csv_path, 'w', encoding='utf-8') as f:
        f.write(header + '\n')
        f.write('\n'.join(rows))
        f.write('\n')

    return [confusion_path, class_wise_path, class_wise_csv_path]


def find_trained_fold_dirs(root='./Kfold_models'):
    if not os.path.isdir(root):
        return []

    fold_dirs = []
    for name in os.listdir(root):
        if not name.startswith('fold'):
            continue
        suffix = name[len('fold'):]
        if not suffix.isdigit():
            continue
        fold = int(suffix)
        fold_dir = os.path.join(root, name)
        if os.path.exists(os.path.join(fold_dir, 'model.pkl')):
            fold_dirs.append((fold, fold_dir))

    return sorted(fold_dirs, key=lambda item: item[0])


def split_window_identity(window_meta, indices):
    selected = [window_meta[int(index)] for index in indices]
    sample_ids = np.asarray([meta['sample_id'] for meta in selected], dtype=str)
    subject_ids = np.asarray([meta['subject_id'] for meta in selected], dtype=str)
    center_epoch_indices = np.asarray([int(meta['center_epoch_index']) for meta in selected], dtype=np.int64)
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


def load_split_metadata(fold, fold_dir, config, dataset, labels, groups=None, window_meta=None, val_window_meta=None):
    metadata_path = os.path.join(fold_dir, SPLIT_METADATA_FILE)
    if not os.path.exists(metadata_path):
        raise RuntimeError(
            f'[ERROR] fold {fold} is missing {SPLIT_METADATA_FILE}. '
            'Refusing to evaluate because the trained test split cannot be verified.'
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
        })
    if dataset.dim() == 5:
        if window_meta is None:
            raise RuntimeError(f'[ERROR] fold {fold} cannot validate context split identity without window metadata.')
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
        raise RuntimeError(f'[ERROR] fold {fold} split metadata is missing keys: {missing_keys}')

    metadata_fold = int(metadata['fold_index'])
    metadata_num_fold = int(metadata['num_fold'])
    metadata_val_ratio = float(metadata['val_ratio'])
    metadata_dataset_shape = tuple(int(v) for v in metadata['dataset_shape'])
    metadata_labels_shape = tuple(int(v) for v in metadata['labels_shape'])
    metadata_train_test_dataset_shape = tuple(int(v) for v in metadata['train_test_dataset_shape'])
    metadata_train_test_labels_shape = tuple(int(v) for v in metadata['train_test_labels_shape'])
    train_idx = np.asarray(metadata['train_idx'], dtype=np.int64)
    test_idx = np.asarray(metadata['test_idx'], dtype=np.int64)

    if metadata_fold != fold:
        raise RuntimeError(f'[ERROR] fold directory fold{fold} contains metadata for fold{metadata_fold}.')
    if metadata_num_fold != config.num_fold:
        raise RuntimeError(
            f'[ERROR] fold {fold} was trained with num_fold={metadata_num_fold}, '
            f'but current config.num_fold={config.num_fold}. Refusing to evaluate mixed split definitions.'
        )
    if int(metadata['random_state']) != SPLIT_RANDOM_STATE:
        raise RuntimeError(f'[ERROR] fold {fold} random_state mismatch in split metadata.')
    if bool(metadata['shuffle']) != SPLIT_SHUFFLE:
        raise RuntimeError(f'[ERROR] fold {fold} shuffle mismatch in split metadata.')
    if not np.isclose(metadata_val_ratio, config.val_ratio):
        raise RuntimeError(
            f'[ERROR] fold {fold} was trained with val_ratio={metadata_val_ratio}, '
            f'but current config.val_ratio={config.val_ratio}. Refusing to evaluate mixed split definitions.'
        )
    if metadata_dataset_shape != tuple(dataset.shape):
        raise RuntimeError(
            f'[ERROR] fold {fold} dataset shape mismatch: trained={metadata_dataset_shape}, current={tuple(dataset.shape)}.'
        )
    if metadata_labels_shape != tuple(labels.shape):
        raise RuntimeError(
            f'[ERROR] fold {fold} labels shape mismatch: trained={metadata_labels_shape}, current={tuple(labels.shape)}.'
        )
    if metadata_train_test_dataset_shape != tuple(dataset.shape):
        raise RuntimeError(
            f'[ERROR] fold {fold} train_test dataset shape mismatch: '
            f'trained={metadata_train_test_dataset_shape}, current={tuple(dataset.shape)}.'
        )
    if metadata_train_test_labels_shape != tuple(labels.shape):
        raise RuntimeError(
            f'[ERROR] fold {fold} train_test labels shape mismatch: '
            f'trained={metadata_train_test_labels_shape}, current={tuple(labels.shape)}.'
        )
    if len(train_idx) == 0 or np.any(train_idx < 0) or np.any(train_idx >= len(labels)):
        raise RuntimeError(f'[ERROR] fold {fold} metadata has invalid train_idx bounds.')
    if len(test_idx) == 0 or np.any(test_idx < 0) or np.any(test_idx >= len(labels)):
        raise RuntimeError(f'[ERROR] fold {fold} metadata has invalid test_idx bounds.')
    if np.intersect1d(train_idx, test_idx).size > 0:
        raise RuntimeError(f'[ERROR] fold {fold} metadata has overlapping train_idx and test_idx.')

    if groups is not None:
        groups = np.asarray(groups)
        train_group_ids, test_group_ids, validation_group_ids = assert_group_split_integrity(
            train_idx,
            test_idx,
            groups,
            val_window_meta=val_window_meta,
            split_name=f'fold {fold}',
        )
        if not np.array_equal(metadata['groups_shape'], np.asarray(groups.shape, dtype=np.int64)):
            raise RuntimeError(f'[ERROR] fold {fold} groups shape mismatch in split metadata.')
        if not np.array_equal(metadata['train_groups'].astype(str), groups[train_idx].astype(str)):
            raise RuntimeError(f'[ERROR] fold {fold} train group identity mismatch in split metadata.')
        if not np.array_equal(metadata['test_groups'].astype(str), groups[test_idx].astype(str)):
            raise RuntimeError(f'[ERROR] fold {fold} test group identity mismatch in split metadata.')
        if not np.array_equal(metadata['train_group_ids'].astype(str), train_group_ids):
            raise RuntimeError(f'[ERROR] fold {fold} train group IDs mismatch in split metadata.')
        if not np.array_equal(metadata['test_group_ids'].astype(str), test_group_ids):
            raise RuntimeError(f'[ERROR] fold {fold} test group IDs mismatch in split metadata.')
        if not np.array_equal(metadata['validation_group_ids'].astype(str), validation_group_ids):
            raise RuntimeError(f'[ERROR] fold {fold} validation group IDs mismatch in split metadata.')
        if str(np.asarray(metadata['group_granularity']).item()) != GROUP_GRANULARITY:
            raise RuntimeError(f'[ERROR] fold {fold} group granularity mismatch in split metadata.')
        if bool(metadata['group_split_enforced']) is not True:
            raise RuntimeError(f'[ERROR] fold {fold} group split flag mismatch in split metadata.')

    if dataset.dim() == 5:
        train_sample_ids, train_subject_ids, train_center_epoch_indices, train_window_labels = split_window_identity(
            window_meta,
            train_idx
        )
        test_sample_ids, test_subject_ids, test_center_epoch_indices, test_window_labels = split_window_identity(
            window_meta,
            test_idx
        )
        context_checks = {
            'context_size': int(metadata['context_size']) == CONTEXT_SIZE,
            'left_context': int(metadata['left_context']) == LEFT_CONTEXT,
            'right_context': int(metadata['right_context']) == RIGHT_CONTEXT,
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
        }
        failed_context_checks = [name for name, passed in context_checks.items() if not passed]
        if failed_context_checks:
            raise RuntimeError(
                f'[ERROR] fold {fold} context split identity mismatch: {", ".join(failed_context_checks)}.'
            )

    return test_idx


def test(model, test_loader, config):
    model.eval()

    pred = []
    label = []

    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Testing', leave=False):
            data = data.to(config.device, non_blocking=True)
            target = target.to(config.device, non_blocking=True)

            output = model(data)
            pred.extend(torch.argmax(output, dim=1).cpu().numpy())
            label.extend(target.cpu().numpy())

    accuracy = accuracy_score(label, pred)
    cohens_kappa = cohen_kappa_score(label, pred)
    macro_f1 = f1_score(label, pred, average='macro')
    weighted_f1 = f1_score(label, pred, average='weighted')
    average_sensitivity = recall_score(label, pred, average='macro')
    average_specificity = specificity(label, pred, n=5)
    balanced_acc = balanced_accuracy_score(label, pred)

    con_mat = confusion_matrix(label, pred, labels=[0, 1, 2, 3, 4])
    class_wise_result = class_wise_evaluate(con_mat)

    print(
        'ACC: %.4f | Kappa: %.4f | Macro-F1: %.4f | Weighted-F1: %.4f | Sens: %.4f | Spec: %.4f | Bal_ACC: %.4f'
        % (accuracy, cohens_kappa, macro_f1, weighted_f1, average_sensitivity, average_specificity, balanced_acc)
    )

    print_class_wise_result(class_wise_result)

    return (
        accuracy,
        cohens_kappa,
        macro_f1,
        weighted_f1,
        average_sensitivity,
        average_specificity,
        balanced_acc,
        con_mat,
        class_wise_result,
    )


def evaluate_single_fold(config, dataset, labels, fold, test_idx):
    path_model = f'./Kfold_models/fold{fold}/model.pkl'
    if not os.path.exists(path_model):
        raise FileNotFoundError(f'Model not found: {path_model}')

    X_test = dataset[test_idx]
    y_test = labels[test_idx]

    test_set = TensorDataset(X_test, y_test)
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )

    model = Transformer(config).to(config.device)
    model.load_state_dict(torch.load(path_model, map_location=config.device), strict=True)

    result = test(model, test_loader, config)

    del model
    torch.cuda.empty_cache()

    return result


def evaluate(config, path, tracker=None):
    if tracker is None:
        tracker = MLflowTracker(config)

    dataset, labels, groups, window_meta, _, val_window_meta = data_generator(
        path_labels=path.path_labels,
        path_dataset=path.path_TF
    )

    if dataset.dim() == 5:
        raise RuntimeError(
            '[ERROR] Context windows were built with shape [N, 5, 3, 29, 128], '
            'but evaluate.py still uses the current Transformer expecting [N, 3, 29, 128]. '
            'Milestone 5 must update model.py before context evaluation can run.'
        )

    with tracker.start_run(run_name=config.mlflow_run_name or 'evaluate') as _:
        tracker.log_params({
            'num_fold': config.num_fold,
            'val_ratio': config.val_ratio,
            'num_classes': config.num_classes,
            'batch_size': config.batch_size,
            'dataset_shape': tuple(dataset.shape),
            'labels_shape': tuple(labels.shape),
            'train_test_dataset_shape': tuple(dataset.shape),
            'train_test_labels_shape': tuple(labels.shape),
            'groups_shape': tuple(groups.shape),
            'window_meta_rows': len(window_meta),
            'val_window_meta_rows': len(val_window_meta),
            'dataset_size': len(labels),
            'device': config.device,
        })

        ACC = 0.0
        Kappa = 0.0
        MF1 = 0.0
        WF1 = 0.0
        Sens = 0.0
        Spec = 0.0
        Bal_ACC = 0.0
        Confusion_mat = np.zeros([5, 5], dtype=np.float64)

        valid_folds = []

        trained_folds = find_trained_fold_dirs()
        if len(trained_folds) == 0:
            raise RuntimeError('[ERROR] No trained fold models found.')

        for fold, fold_dir in trained_folds:
            test_idx = load_split_metadata(
                fold,
                fold_dir,
                config,
                dataset,
                labels,
                groups,
                window_meta,
                val_window_meta,
            )

            print('\n' + '-' * 15, '>', f'Fold {fold}', '<', '-' * 15)

            (
                accuracy,
                cohens_kappa,
                macro_f1,
                weighted_f1,
                average_sensitivity,
                average_specificity,
                balanced_acc,
                con_mat,
                class_wise_fold
            ) = evaluate_single_fold(config, dataset, labels, fold, test_idx)

            tracker.log_metrics({
                f'fold_{fold}_acc': accuracy,
                f'fold_{fold}_kappa': cohens_kappa,
                f'fold_{fold}_macro_f1': macro_f1,
                f'fold_{fold}_weighted_f1': weighted_f1,
                f'fold_{fold}_sensitivity_macro_recall': average_sensitivity,
                f'fold_{fold}_specificity': average_specificity,
                f'fold_{fold}_balanced_accuracy': balanced_acc,
            })

            fold_artifact_dir = f'./Kfold_models/fold{fold}/evaluation'
            fold_artifacts = save_evaluation_artifacts(con_mat, class_wise_fold, output_dir=fold_artifact_dir)
            for artifact_path in fold_artifacts:
                tracker.log_artifact(artifact_path, artifact_path=f'fold{fold}/evaluation')

            ACC += accuracy
            Kappa += cohens_kappa
            MF1 += macro_f1
            WF1 += weighted_f1
            Sens += average_sensitivity
            Spec += average_specificity
            Bal_ACC += balanced_acc
            Confusion_mat += con_mat

            valid_folds.append(fold)

        if len(valid_folds) == 0:
            raise RuntimeError('[ERROR] No trained fold models found.')

        num_valid = len(valid_folds)
        ACC /= num_valid
        Kappa /= num_valid
        MF1 /= num_valid
        WF1 /= num_valid
        Sens /= num_valid
        Spec /= num_valid
        Bal_ACC /= num_valid

        class_wise_result = class_wise_evaluate(Confusion_mat)

        tracker.log_metrics({
            'mean_acc': ACC,
            'mean_kappa': Kappa,
            'mean_macro_f1': MF1,
            'mean_weighted_f1': WF1,
            'mean_sensitivity_macro_recall': Sens,
            'mean_specificity': Spec,
            'mean_balanced_accuracy': Bal_ACC,
            'num_valid_folds': num_valid,
        })

        artifact_paths = save_evaluation_artifacts(Confusion_mat, class_wise_result)
        for artifact_path in artifact_paths:
            tracker.log_artifact(artifact_path, artifact_path='evaluation')

        return ACC, Kappa, MF1, WF1, Sens, Spec, Bal_ACC, Confusion_mat, class_wise_result, valid_folds


if __name__ == '__main__':
    config = Config()
    path = Path()

    ACC, Kappa, MF1, WF1, Sens, Spec, Bal_ACC, Confusion_mat, class_wise_result, valid_folds = evaluate(config, path)

    print('\n===== FINAL RESULT =====')
    print('valid_folds: ', valid_folds)
    print('ACC: ', ACC)
    print("Cohen's Kappa: ", Kappa)
    print('Macro-F1: ', MF1)
    print('Weighted-F1: ', WF1)
    print('Sensitivity (Macro Recall): ', Sens)
    print('Specificity (Macro): ', Spec)
    print('Balanced Accuracy: ', Bal_ACC)

    print('\nconfusion_mat:')
    print(Confusion_mat)

    print_class_wise_result(class_wise_result)
