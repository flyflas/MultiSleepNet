import os

import numpy as np


SPLIT_METADATA_FILE = 'split_metadata.npz'
CHECKPOINT_METADATA_FILE = 'checkpoint_metadata.npz'
SPLIT_RANDOM_STATE = 0
SPLIT_SHUFFLE = True
GROUP_GRANULARITY = 'subject_id'
DATA_GENERATOR_INTERFACE_VERSION = 2
MODEL_VARIANT_CONTEXT = 'context'
MODEL_VARIANT_LEGACY = 'legacy'
NORMALIZATION_STRATEGY = 'global_channel_mean_std'
FUSION_BOUNDARY_SHAPE = 'sequence_tf_tokens'


def checkpoint_root(config):
    return getattr(config, 'variant_checkpoint_root', os.path.join('./Kfold_models', config.model_variant))


def fold_dir(config, fold):
    return os.path.join(checkpoint_root(config), f'fold{fold}')


def evaluation_dir(config):
    return os.path.join(checkpoint_root(config), 'evaluation')


def model_path(config, fold):
    return os.path.join(fold_dir(config, fold), 'model.pkl')


def checkpoint_metadata_path(config, fold):
    return os.path.join(fold_dir(config, fold), CHECKPOINT_METADATA_FILE)


def split_metadata_path(fold_path):
    return os.path.join(fold_path, SPLIT_METADATA_FILE)


def expected_input_shape(dataset):
    return np.asarray(tuple(dataset.shape[1:]), dtype=np.int64)


def expected_fusion_boundary_shape(config):
    return np.asarray((config.tf_seq_len, config.dim_model), dtype=np.int64)


def expected_model_variant_for_dataset(dataset):
    if dataset.dim() == 5:
        return MODEL_VARIANT_CONTEXT
    if dataset.dim() == 4:
        return MODEL_VARIANT_LEGACY
    raise RuntimeError(f'[ERROR] Unsupported dataset rank {dataset.dim()} for compatibility metadata.')


def scalar_string(value):
    return str(np.asarray(value).item())


def scalar_int(value):
    return int(np.asarray(value).item())


def scalar_bool(value):
    return bool(np.asarray(value).item())


def current_compatibility_metadata(config, dataset):
    return {
        'input_shape': expected_input_shape(dataset),
        'context_size': np.array(config.context_size, dtype=np.int64),
        'left_context': np.array(config.left_context, dtype=np.int64),
        'right_context': np.array(config.right_context, dtype=np.int64),
        'group_granularity': np.array(GROUP_GRANULARITY),
        'normalization_strategy': np.array(config.normalization_strategy),
        'model_variant': np.array(config.model_variant),
        'data_generator_interface_version': np.array(config.data_generator_interface_version, dtype=np.int64),
        'tf_seq_len': np.array(config.tf_seq_len, dtype=np.int64),
        'fusion_boundary_shape': expected_fusion_boundary_shape(config),
        'fusion_boundary_type': np.array(config.fusion_boundary_shape),
    }


def required_compatibility_keys():
    return {
        'input_shape',
        'context_size',
        'left_context',
        'right_context',
        'group_granularity',
        'group_ids_train',
        'group_ids_test',
        'group_ids_val',
        'normalization_strategy',
        'model_variant',
        'data_generator_interface_version',
        'tf_seq_len',
        'fusion_boundary_shape',
        'fusion_boundary_type',
    }


def build_checkpoint_metadata(config, dataset, fold, train_group_ids, test_group_ids, validation_group_ids):
    metadata = current_compatibility_metadata(config, dataset)
    metadata.update({
        'fold_index': np.array(fold, dtype=np.int64),
        'checkpoint_schema_version': np.array(1, dtype=np.int64),
        'group_ids_train': np.asarray(train_group_ids, dtype=str),
        'group_ids_test': np.asarray(test_group_ids, dtype=str),
        'group_ids_val': np.asarray(validation_group_ids, dtype=str),
    })
    return metadata


def assert_runtime_compatibility(config, dataset):
    expected_variant = expected_model_variant_for_dataset(dataset)
    if config.model_variant != expected_variant:
        raise RuntimeError(
            f'[ERROR] Config model_variant={config.model_variant!r} is incompatible with dataset rank '
            f'{dataset.dim()} and input shape {tuple(dataset.shape[1:])}; expected {expected_variant!r}.'
        )
    if config.data_generator_interface_version != DATA_GENERATOR_INTERFACE_VERSION:
        raise RuntimeError(
            '[ERROR] Incompatible data generator interface version: '
            f'config={config.data_generator_interface_version}, expected={DATA_GENERATOR_INTERFACE_VERSION}.'
        )
    if config.normalization_strategy != NORMALIZATION_STRATEGY:
        raise RuntimeError(
            f'[ERROR] normalization_strategy={config.normalization_strategy!r} is not supported by this pipeline; '
            f'expected {NORMALIZATION_STRATEGY!r}.'
        )
    if config.fusion_boundary_shape != FUSION_BOUNDARY_SHAPE:
        raise RuntimeError(
            f'[ERROR] fusion_boundary_shape={config.fusion_boundary_shape!r} is not supported; '
            f'expected {FUSION_BOUNDARY_SHAPE!r}.'
        )


def assert_metadata_compatibility(metadata, config, dataset, fold, context):
    missing_keys = sorted(required_compatibility_keys() - set(metadata.files))
    if missing_keys:
        raise RuntimeError(
            f'[ERROR] fold {fold} {context} metadata is legacy or incomplete; missing compatibility keys: '
            f'{missing_keys}. Refusing to mix old artifacts with {config.model_variant!r} model evaluation.'
        )

    expected = current_compatibility_metadata(config, dataset)
    checks = {
        'input_shape': np.array_equal(metadata['input_shape'], expected['input_shape']),
        'context_size': scalar_int(metadata['context_size']) == config.context_size,
        'left_context': scalar_int(metadata['left_context']) == config.left_context,
        'right_context': scalar_int(metadata['right_context']) == config.right_context,
        'group_granularity': scalar_string(metadata['group_granularity']) == GROUP_GRANULARITY,
        'normalization_strategy': (
            scalar_string(metadata['normalization_strategy']) == config.normalization_strategy
        ),
        'model_variant': scalar_string(metadata['model_variant']) == config.model_variant,
        'data_generator_interface_version': (
            scalar_int(metadata['data_generator_interface_version']) == config.data_generator_interface_version
        ),
        'tf_seq_len': scalar_int(metadata['tf_seq_len']) == config.tf_seq_len,
        'fusion_boundary_shape': np.array_equal(
            metadata['fusion_boundary_shape'],
            expected['fusion_boundary_shape'],
        ),
        'fusion_boundary_type': scalar_string(metadata['fusion_boundary_type']) == config.fusion_boundary_shape,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            f'[ERROR] fold {fold} {context} compatibility mismatch: {", ".join(failed)}. '
            f'Checkpoint/split metadata cannot be used with current model_variant={config.model_variant!r}.'
        )


def assert_checkpoint_compatibility(checkpoint_metadata, split_metadata, config, dataset, fold):
    assert_metadata_compatibility(checkpoint_metadata, config, dataset, fold, 'checkpoint')

    group_checks = {
        'group_ids_train': np.array_equal(
            checkpoint_metadata['group_ids_train'].astype(str),
            split_metadata['group_ids_train'].astype(str),
        ),
        'group_ids_test': np.array_equal(
            checkpoint_metadata['group_ids_test'].astype(str),
            split_metadata['group_ids_test'].astype(str),
        ),
        'group_ids_val': np.array_equal(
            checkpoint_metadata['group_ids_val'].astype(str),
            split_metadata['group_ids_val'].astype(str),
        ),
    }
    failed = [name for name, passed in group_checks.items() if not passed]
    if failed:
        raise RuntimeError(
            f'[ERROR] fold {fold} checkpoint group IDs do not match split metadata: {", ".join(failed)}.'
        )


def save_checkpoint_metadata(config, dataset, fold, train_group_ids, test_group_ids, validation_group_ids):
    metadata_path = checkpoint_metadata_path(config, fold)
    np.savez(
        metadata_path,
        **build_checkpoint_metadata(config, dataset, fold, train_group_ids, test_group_ids, validation_group_ids),
    )
    return metadata_path
