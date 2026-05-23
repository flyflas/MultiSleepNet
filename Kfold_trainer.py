import os
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from model_transformer_cross_c import Transformer
from early_stop_tool import EarlyStopping
from data_loader import data_generator
from args import Config, Path


def set_random_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 为了结果更稳定，先关闭 benchmark
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_label_distribution(labels, split_name='dataset'):
    """Print label distribution of a numpy array or torch tensor."""
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)

    print(f'\n[{split_name}] label distribution:')
    for u, c in zip(unique, counts):
        print(f'  class {u}: {c} ({c / total:.6f})')


def build_dataloader(dataset, batch_size, shuffle, num_workers=8):
    """Build DataLoader with safer worker settings."""
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
    """
    Evaluate model on a dataloader.
    Returns:
        accuracy, avg_loss
    """
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


def train(save_all_checkpoint=False):
    config = Config()
    path = Path()

    print(f'[INFO] device = {config.device}')
    print(f'[INFO] batch_size = {config.batch_size}')
    print(f'[INFO] learning_rate = {config.learning_rate}')
    print(f'[INFO] num_epochs = {config.num_epochs}')

    # 保持你当前项目的调用方式
    dataset, labels, val_loader = data_generator(
        path_labels=path.path_labels,
        path_dataset=path.path_TF
    )

    print(f'[INFO] dataset shape: {dataset.shape}')
    print(f'[INFO] labels shape: {labels.shape}')
    print_label_distribution(labels, split_name='full dataset')

    kf = StratifiedKFold(
        n_splits=config.num_fold,
        shuffle=True,
        random_state=0
    )

    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset, labels)):
        print('\n' + '-' * 15 + f' > Fold {fold} < ' + '-' * 15)

        fold_dir = f'./Kfold_models/fold{fold}'
        os.makedirs(fold_dir, exist_ok=True)

        X_train, X_test = dataset[train_idx], dataset[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        print(f'[INFO][fold {fold}] X_train shape = {X_train.shape}, y_train shape = {y_train.shape}')
        print(f'[INFO][fold {fold}] X_test  shape = {X_test.shape}, y_test  shape = {y_test.shape}')

        print_label_distribution(y_train, split_name=f'fold {fold} train')
        print_label_distribution(y_test, split_name=f'fold {fold} test')

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

        # 保持你目前的优化器设置
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

            # 只在前 3 个 epoch 和每 10 个 epoch 打印一次分布，避免日志太长
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

            # 保持你当前早停调用方式
            model_path = os.path.join(fold_dir, f'model_{fold}_epoch{epoch}.pkl')
            early_stopping(val_acc, model, path=model_path)

            if early_stopping.early_stop:
                print(f'[INFO] Early stopping at epoch {epoch}')
                break

        np.save(os.path.join(fold_dir, 'train_LOSS.npy'), np.array(train_LOSS))
        np.save(os.path.join(fold_dir, 'train_ACC.npy'), np.array(train_ACC))
        np.save(os.path.join(fold_dir, 'test_LOSS.npy'), np.array(test_LOSS))
        np.save(os.path.join(fold_dir, 'test_ACC.npy'), np.array(test_ACC))
        np.save(os.path.join(fold_dir, 'val_LOSS.npy'), np.array(val_LOSS))
        np.save(os.path.join(fold_dir, 'val_ACC.npy'), np.array(val_ACC))

        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    set_random_seed(0)
    train(save_all_checkpoint=False)
    
    

