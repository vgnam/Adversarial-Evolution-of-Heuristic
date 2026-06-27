"""Helpers for loading fixed benchmark datasets cached on disk.

The fixed-dataset workflow uses one default training dataset per task plus
optional fixed-size train/test datasets:

    <task_folder>/train_dataset.pkl
    <task_folder>/train_datasets/size_<N>.pkl
    <task_folder>/test_datasets/size_<N>.pkl

For backward compatibility, ``load_dataset_file(..., split='train')`` falls
back to the previous ``dataset.pkl`` name when ``train_dataset.pkl`` is not
present.
"""
from __future__ import annotations

import os
import pickle
import re
from typing import Any, Dict, Iterable, List


TRAIN_DATASET_FILENAME = 'train_dataset.pkl'
LEGACY_DATASET_FILENAME = 'dataset.pkl'
TRAIN_DATASET_DIRNAME = 'train_datasets'
TEST_DATASET_DIRNAME = 'test_datasets'
SIZED_DATASET_PATTERN = re.compile(r'^size_(\d+)\.pkl$')


def train_dataset_filename(size: int) -> str:
    """Return the relative filename for a train dataset of a given size."""
    return os.path.join(TRAIN_DATASET_DIRNAME, f'size_{size}.pkl')


def test_dataset_filename(size: int) -> str:
    """Return the relative filename for a test dataset of a given size."""
    return os.path.join(TEST_DATASET_DIRNAME, f'size_{size}.pkl')


def _candidate_paths(task_folder: str,
                     filename: str | None,
                     split: str,
                     size: int | None,
                     allow_legacy_train: bool) -> List[str]:
    if filename is not None:
        return [filename if os.path.isabs(filename) else os.path.join(task_folder, filename)]

    if split == 'train':
        if size is not None:
            return [os.path.join(task_folder, train_dataset_filename(size))]
        filenames = [TRAIN_DATASET_FILENAME]
        if allow_legacy_train:
            filenames.append(LEGACY_DATASET_FILENAME)
        return [os.path.join(task_folder, name) for name in filenames]

    if split == 'test':
        if size is None:
            raise ValueError("dataset_size is required when loading split='test'.")
        return [os.path.join(task_folder, test_dataset_filename(size))]

    raise ValueError("split must be either 'train' or 'test'.")


def load_dataset_file(task_folder: str,
                      filename: str | None = None,
                      *,
                      split: str = 'train',
                      size: int | None = None,
                      allow_legacy_train: bool = True) -> List[Any]:
    """Load a pickled dataset list from a task folder.

    Args:
        task_folder: task directory containing ``evaluation.py``.
        filename: optional explicit pickle filename or absolute path.
        split: ``'train'`` loads ``train_dataset.pkl`` when size is omitted,
            or ``train_datasets/size_<size>.pkl`` when size is provided.
            ``'test'`` loads
            ``test_datasets/size_<size>.pkl``.
        size: problem size used in fixed-size train/test dataset filenames.
        allow_legacy_train: when loading train data, also try ``dataset.pkl``.
    """
    paths = _candidate_paths(task_folder, filename, split, size, allow_legacy_train)
    path = next((candidate for candidate in paths if os.path.isfile(candidate)), None)

    if path is None:
        expected = '\n'.join(f"    {candidate}" for candidate in paths)
        task_name = os.path.basename(task_folder)
        hint = f"python llm4ad/task/optimization/generate_fixed_datasets.py --task {task_name}"
        if split == 'train' and size is not None:
            hint += f" --split train --train-sizes {size}"
        if split == 'test' and size is not None:
            hint += f" --split test --test-sizes {size}"
        raise FileNotFoundError(
            f"Dataset file not found. Expected one of:\n{expected}\n"
            f"Generate it by running:\n"
            f"    {hint}"
        )

    with open(path, 'rb') as f:
        return pickle.load(f)


def _list_sized_dataset_sizes(task_folder: str, dirname: str) -> List[int]:
    dataset_dir = os.path.join(task_folder, dirname)
    if not os.path.isdir(dataset_dir):
        return []

    sizes = []
    for filename in os.listdir(dataset_dir):
        match = SIZED_DATASET_PATTERN.match(filename)
        if match:
            sizes.append(int(match.group(1)))
    return sorted(sizes)


def list_train_dataset_sizes(task_folder: str) -> List[int]:
    """List available fixed train dataset sizes for a task folder."""
    return _list_sized_dataset_sizes(task_folder, TRAIN_DATASET_DIRNAME)


def list_test_dataset_sizes(task_folder: str) -> List[int]:
    """List available fixed test dataset sizes for a task folder."""
    return _list_sized_dataset_sizes(task_folder, TEST_DATASET_DIRNAME)


def load_train_datasets(task_folder: str, sizes: Iterable[int] | None = None) -> Dict[int, List[Any]]:
    """Load multiple train datasets keyed by problem size."""
    requested_sizes = list_train_dataset_sizes(task_folder) if sizes is None else list(sizes)
    return {
        size: load_dataset_file(task_folder, split='train', size=size)
        for size in requested_sizes
    }


def load_test_datasets(task_folder: str, sizes: Iterable[int] | None = None) -> Dict[int, List[Any]]:
    """Load multiple test datasets keyed by problem size."""
    requested_sizes = list_test_dataset_sizes(task_folder) if sizes is None else list(sizes)
    return {
        size: load_dataset_file(task_folder, split='test', size=size)
        for size in requested_sizes
    }
