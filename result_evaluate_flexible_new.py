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

from sklearn.model_selection import StratifiedKFold

import torch
from torch.utils.data import TensorDataset, DataLoader

from model_transformer_cross_c import Transformer
from data_loader import data_generator
from args import Config, Path


CLASS_NAMES = ['Wake', 'N1', 'N2', 'N3', 'REM']


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


def evaluate(config, path):
    dataset, labels, _ = data_generator(
        path_labels=path.path_labels,
        path_dataset=path.path_TF
    )

    kf = StratifiedKFold(n_splits=config.num_fold, shuffle=True, random_state=0)

    ACC = 0.0
    Kappa = 0.0
    MF1 = 0.0
    WF1 = 0.0
    Sens = 0.0
    Spec = 0.0
    Bal_ACC = 0.0
    Confusion_mat = np.zeros([5, 5], dtype=np.float64)

    valid_folds = []

    for fold, (_, test_idx) in enumerate(kf.split(dataset, labels)):
        path_model = f'./Kfold_models/fold{fold}/model.pkl'
        if not os.path.exists(path_model):
            print(f'[SKIP] fold {fold} model not found')
            continue

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
            _
        ) = evaluate_single_fold(config, dataset, labels, fold, test_idx)

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