"""
    根据名称获取数据集
"""
import os
import sys
import json

import medmnist
from torchvision import transforms, datasets
from medmnist import INFO, Evaluator
# from medmnist import INFO
from base import get_transform
from augmentations.transform3d import get_3d_transform

from medmnist import PathMNIST
from medmnist import DermaMNIST
from medmnist import PneumoniaMNIST
from medmnist import TissueMNIST
from medmnist import BloodMNIST
from medmnist import BreastMNIST
# from medmnist import OrganAMNIST
# from medmnist import OrganBMNIST
# from medmnist import OrganCMNIST
from medmnist import INFO


def get_dataset_and_info(data_flag: str, aug: str, resize: bool, root_dir: str):
    print('2')
    dataset = None
    info = {}
    train_transform, test_transform = get_transform(data_flag, aug, resize)
    print('3')
    # building dataset
    train_dataset = datasets.ImageFolder(root=os.path.join(root_dir, "train"), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=os.path.join(root_dir, "test"), transform=test_transform)

    # 假设数据集的info信息已知
    data_info = {
        'task': 'multi-label, binary-class',  # 或其他任务类型
        'n_channels': 3,  # 假设图像是RGB的
        'label': ['label1', 'label2'],  # 根据您的数据集标签进行调整

        'n_samples': {'train': len(train_dataset), 'val': len(val_dataset)},
        'size': 32,  # 图像尺寸
    }
    # print('1')
    # print(f"train_dataset: {train_dataset}, val_dataset: {val_dataset}, data_info: {data_info}")
    return (train_dataset, val_dataset), data_info
    # test_dataset = datasets.ImageFolder(root=os.path.join(root_dir, "test"), transform=test_transform)
    # 'n_samples': {'train': len(train_dataset), 'val': len(val_dataset), 'test': len(test_dataset)},
    # return (train_dataset, val_dataset, test_dataset), data_info


def get_3d_dataset(data_flag: str, train_transform, test_transform):
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    train_dataset = DataClass(split='train', download=True, transform=train_transform, size=64)
    val_dataset = DataClass(split='val', transform=test_transform, download=True, size=64)
    test_dataset = DataClass(split='test', transform=test_transform, download=True, size=64)

    data_info = {}
    data_info['task'] = info['task']
    data_info['n_channels'] = info['n_channels']
    data_info['label'] = info['label']
    data_info['n_samples'] = info['n_samples']
    data_info['size'] = 64

    data_info['train_evaluator'] = Evaluator(data_flag, 'train', size=64)
    data_info['val_evaluator'] = Evaluator(data_flag, 'val', size=64)
    data_info['test_evaluator'] = Evaluator(data_flag, 'test', size=64)

    return (train_dataset, val_dataset, test_dataset), data_info


def get_medmnist_dataset(data_flag: str, train_transform, test_transform):
    download_medmnist(data_flag, 224)
    print('2')
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    train_dataset = DataClass(split='train', download=True, transform=train_transform, size=224)
    val_dataset = DataClass(split='val', transform=test_transform, download=True, size=224)
    test_dataset = DataClass(split='test', transform=test_transform, download=True, size=224)

    data_info = {}
    data_info['task'] = info['task']
    data_info['n_channels'] = info['n_channels']
    data_info['label'] = info['label']
    data_info['n_samples'] = info['n_samples']
    data_info['size'] = 224

    data_info['train_evaluator'] = Evaluator(data_flag, 'train', size=224)
    data_info['val_evaluator'] = Evaluator(data_flag, 'val', size=224)
    data_info['test_evaluator'] = Evaluator(data_flag, 'test', size=224)

    return (train_dataset, val_dataset), data_info


def download_medmnist(data_flag: str, size: int):
    info = INFO[data_flag]
    print(data_flag)
    DataClass = getattr(medmnist, info['python_class'])
    filename = f"{data_flag}_{size}.npz"
    url = f"https://zenodo.org/record/10519652/files/{filename}?download=1"
    # url = f"https://zenodo.org/record/10519652/files/{filename}?download=1"
    download_path = os.path.join(os.path.expanduser("~"), '.medmnist', filename)

    if not os.path.exists(download_path):
        print(f"Downloading {data_flag} dataset...")
        os.makedirs(os.path.dirname(download_path), exist_ok=True)
        import requests
        response = requests.get(url)
        with open(download_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {data_flag} dataset to {download_path}")
    else:
        print(f"{data_flag} dataset already exists.")