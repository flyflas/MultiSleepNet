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


def load_split_metadata(fold, fold_dir, config, tf_dataset, sse_dataset, labels):
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
        'tf_dataset_shape',
        'sse_dataset_shape',
        'labels_shape',
        'train_test_dataset_shape',
        'train_test_labels_shape',
    }
    missing_keys = sorted(required_keys - set(metadata.files))
    if missing_keys:
        raise RuntimeError(f'[ERROR] fold {fold} split metadata is missing keys: {missing_keys}')

    metadata_fold = int(metadata['fold_index'])
    metadata_num_fold = int(metadata['num_fold'])
    metadata_val_ratio = float(metadata['val_ratio'])
    metadata_tf_dataset_shape = tuple(int(v) for v in metadata['tf_dataset_shape'])
    metadata_sse_dataset_shape = tuple(int(v) for v in metadata['sse_dataset_shape'])
    metadata_labels_shape = tuple(int(v) for v in metadata['labels_shape'])
    metadata_train_test_tf_dataset_shape = tuple(int(v) for v in metadata['train_test_tf_dataset_shape'])
    metadata_train_test_sse_dataset_shape = tuple(int(v) for v in metadata['train_test_sse_dataset_shape'])
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
    if metadata_tf_dataset_shape != tuple(tf_dataset.shape):
        raise RuntimeError(
            f'[ERROR] fold {fold} TF dataset shape mismatch: trained={metadata_tf_dataset_shape}, current={tuple(tf_dataset.shape)}.'
        )
    if metadata_sse_dataset_shape != tuple(sse_dataset.shape):
        raise RuntimeError(
            f'[ERROR] fold {fold} SSE dataset shape mismatch: trained={metadata_sse_dataset_shape}, current={tuple(sse_dataset.shape)}.'
        )
    if metadata_labels_shape != tuple(labels.shape):
        raise RuntimeError(
            f'[ERROR] fold {fold} labels shape mismatch: trained={metadata_labels_shape}, current={tuple(labels.shape)}.'
        )
    if metadata_train_test_tf_dataset_shape != tuple(tf_dataset.shape):
        raise RuntimeError(
            f'[ERROR] fold {fold} train_test TF dataset shape mismatch: '
            f'trained={metadata_train_test_tf_dataset_shape}, current={tuple(tf_dataset.shape)}.'
        )
    if metadata_train_test_sse_dataset_shape != tuple(sse_dataset.shape):
        raise RuntimeError(
            f'[ERROR] fold {fold} train_test SSE dataset shape mismatch: '
            f'trained={metadata_train_test_sse_dataset_shape}, current={tuple(sse_dataset.shape)}.'
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

    return test_idx


def test(model, test_loader, config):
    model.eval()

    pred = []
    label = []

    with torch.no_grad():
        for tf_data, sse_data, target in tqdm(test_loader, desc='Testing', leave=False):
            tf_data = tf_data.to(config.device, non_blocking=True)
            sse_data = sse_data.to(config.device, non_blocking=True)
            target = target.to(config.device, non_blocking=True)

            output = model(tf_data, sse_data)
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


def evaluate_single_fold(config, tf_dataset, sse_dataset, labels, fold, test_idx):
    path_model = f'./Kfold_models/fold{fold}/model.pkl'
    if not os.path.exists(path_model):
        raise FileNotFoundError(f'Model not found: {path_model}')

    tf_test = tf_dataset[test_idx]
    sse_test = sse_dataset[test_idx]
    y_test = labels[test_idx]

    test_set = TensorDataset(tf_test, sse_test, y_test)
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

    tf_dataset, sse_dataset, labels, _ = data_generator(
        path_labels=path.path_labels,
        path_dataset=path.path_TF
    )

    with tracker.start_run(run_name=config.mlflow_run_name or 'evaluate') as _:
        tracker.log_params({
            'num_fold': config.num_fold,
            'val_ratio': config.val_ratio,
            'num_classes': config.num_classes,
            'batch_size': config.batch_size,
            'tf_dataset_shape': tuple(tf_dataset.shape),
            'sse_dataset_shape': tuple(sse_dataset.shape),
            'labels_shape': tuple(labels.shape),
            'train_test_tf_dataset_shape': tuple(tf_dataset.shape),
            'train_test_sse_dataset_shape': tuple(sse_dataset.shape),
            'train_test_labels_shape': tuple(labels.shape),
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
            test_idx = load_split_metadata(fold, fold_dir, config, tf_dataset, sse_dataset, labels)

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
            ) = evaluate_single_fold(config, tf_dataset, sse_dataset, labels, fold, test_idx)

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
