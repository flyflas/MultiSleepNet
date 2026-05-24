import csv
import json
import os

import numpy as np
import torch

from compatibility import GROUP_GRANULARITY, checkpoint_root


def count_model_parameters(model):
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _context_epoch_terms(left_context, right_context):
    terms = []
    for offset in range(-left_context, right_context + 1):
        if offset == 0:
            terms.append('t')
        elif offset < 0:
            terms.append(f't{offset}')
        else:
            terms.append(f't+{offset}')
    return terms


def context_window_notation(config):
    terms = _context_epoch_terms(config.left_context, config.right_context)
    if len(terms) == 1:
        return terms[0]
    return f'{terms[0]}..{terms[-1]}'


def future_epoch_terms(config):
    return [f't+{offset}' for offset in range(1, config.right_context + 1)]


def split_description():
    return (
        f'subject-level split using {GROUP_GRANULARITY}; not a PSG-file-level split '
        'and not an epoch-level random split'
    )


def experiment_name(config):
    if config.model_variant == 'context':
        return f'context_{context_window_notation(config)}_subject_split'
    return 'legacy_single_epoch_subject_split'


def _to_jsonable(value):
    if isinstance(value, torch.Size):
        return list(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    return value


def build_experiment_metadata(
    config,
    dataset,
    *,
    stage,
    model_parameter_count=None,
    fold=None,
    metrics=None,
    sizes=None,
):
    input_shape = tuple(dataset.shape[1:]) if hasattr(dataset, 'shape') else None
    uses_future_epochs = config.right_context > 0
    is_context = config.model_variant == 'context'
    task_description = (
        '5-epoch context model centered on label[t]'
        if is_context else
        'legacy single-epoch baseline centered on label[t]'
    )

    metadata = {
        'experiment_name': experiment_name(config),
        'stage': stage,
        'fold_index': fold,
        'model_variant': config.model_variant,
        'task_description': task_description,
        'legacy_baseline': not is_context,
        'single_epoch_baseline': not is_context,
        'context_size': int(config.context_size),
        'left_context': int(config.left_context),
        'right_context': int(config.right_context),
        'context_window': context_window_notation(config),
        'context_window_epochs': _context_epoch_terms(config.left_context, config.right_context),
        'label_epoch': 't',
        'uses_future_epochs': bool(uses_future_epochs),
        'future_epochs_used': future_epoch_terms(config),
        'split_granularity': GROUP_GRANULARITY,
        'split_level': 'subject-level',
        'psg_file_level_split': False,
        'epoch_level_random_split': False,
        'split_description': split_description(),
        'normalization_strategy': config.normalization_strategy,
        'batch_size': int(config.batch_size),
        'input_shape': input_shape,
        'model_parameter_count': model_parameter_count,
        'num_fold': int(config.num_fold),
        'val_ratio': float(config.val_ratio),
        'checkpoint_root': checkpoint_root(config),
        'metrics_scope_warning': (
            'These metrics come from subject-level grouped folds for the reported model_variant; '
            'do not compare them as legacy epoch-level random split metrics.'
        ),
    }
    if sizes:
        metadata.update(sizes)
    if metrics:
        metadata['metrics'] = metrics
    return _to_jsonable(metadata)


def write_experiment_report(output_dir, metadata, prefix='experiment_report'):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f'{prefix}.json')
    txt_path = os.path.join(output_dir, f'{prefix}.txt')

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write('\n')

    lines = [
        f'Experiment: {metadata["experiment_name"]}',
        f'Stage: {metadata["stage"]}',
        f'Model variant: {metadata["model_variant"]}',
        f'Task: {metadata["task_description"]}',
        f'Context window: {metadata["context_window"]}',
        f'Context epochs: {", ".join(metadata["context_window_epochs"])}',
        f'Uses future epochs: {metadata["uses_future_epochs"]}',
        f'Future epochs used: {", ".join(metadata["future_epochs_used"]) or "none"}',
        f'Split level: {metadata["split_level"]}',
        f'Split granularity: {metadata["split_granularity"]}',
        f'PSG-file-level split: {metadata["psg_file_level_split"]}',
        f'Epoch-level random split: {metadata["epoch_level_random_split"]}',
        f'Normalization: {metadata["normalization_strategy"]}',
        f'Batch size: {metadata["batch_size"]}',
        f'Input shape: {metadata["input_shape"]}',
        f'Model parameter count: {metadata["model_parameter_count"]}',
        f'Warning: {metadata["metrics_scope_warning"]}',
    ]
    if metadata.get('metrics'):
        lines.append('Metrics:')
        for key, value in metadata['metrics'].items():
            lines.append(f'  {key}: {value}')

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
        f.write('\n')

    return [json_path, txt_path]


def write_metrics_summary_csv(output_dir, rows, filename='metrics_summary.csv'):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fieldnames = [
        'scope',
        'fold',
        'model_variant',
        'experiment_name',
        'context_window',
        'uses_future_epochs',
        'split_level',
        'split_granularity',
        'normalization_strategy',
        'batch_size',
        'input_shape',
        'model_parameter_count',
        'accuracy',
        'kappa',
        'macro_f1',
        'weighted_f1',
        'sensitivity_macro_recall',
        'specificity',
        'balanced_accuracy',
        'metrics_scope_warning',
    ]
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, '') for field in fieldnames})
    return path
