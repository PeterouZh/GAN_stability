import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np


def get_dataset(name, data_dir, size=64, lsun_categories=None, load_in_mem=False):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Lambda(lambda x: x + 1./128 * torch.rand(x.size())),
    ])
    data_dir = os.path.expanduser(data_dir)
    if name == 'image':
        dataset = datasets.ImageFolder(data_dir, transform)
        nlabels = len(dataset.classes)
    elif name == 'hdf5':
        from TOOLS.make_hdf5 import Dataset_HDF5
        transform = transforms.Compose([
            transforms.Lambda(lambda x: x.transpose(1, 2, 0)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Lambda(lambda x: x + 1. / 128 * torch.rand(x.size())),
        ])
        dataset = Dataset_HDF5(root=data_dir, transform=transform,
                               load_in_mem=load_in_mem)
        nlabels = len(dataset.classes)
    elif name == 'npy':
        # Only support normalization for now
        dataset = datasets.DatasetFolder(data_dir, npy_loader, ['npy'])
        nlabels = len(dataset.classes)
    elif name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = datasets.CIFAR10(root=data_dir, train=True, download=True,
                                   transform=transform)
        nlabels = 10
    elif name == 'lsun':
        if lsun_categories is None:
            lsun_categories = 'train'
        dataset = datasets.LSUN(data_dir, lsun_categories, transform)
        nlabels = len(dataset.classes)
    elif name == 'lsun_class':
        dataset = datasets.LSUNClass(data_dir, transform,
                                     target_transform=(lambda t: 0))
        nlabels = 1
    else:
        raise NotImplemented

    return dataset, nlabels


def npy_loader(path):
    img = np.load(path)

    if img.dtype == np.uint8:
        img = img.astype(np.float32)
        img = img/127.5 - 1.
    elif img.dtype == np.float32:
        img = img * 2 - 1.
    else:
        raise NotImplementedError

    img = torch.Tensor(img)
    if len(img.size()) == 4:
        img.squeeze_(0)

    return img
