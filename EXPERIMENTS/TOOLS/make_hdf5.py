import torch.utils.data
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from collections import OrderedDict
from pprint import pformat
import os
import sys
from argparse import ArgumentParser

import unittest
from tqdm import tqdm, trange
import h5py as h5


def make_hdf5(dataloader, config, myargs):
  # Update compression entry
  # No compression; can also use 'lzf'
  config['compression'] = 'lzf' if config['compression'] else None
  config.saved_hdf5 = os.path.expanduser(config.saved_hdf5)
  print('Starting to load into an HDF5 file with chunk size %i '
        'and compression %s...'
        % (config['chunk_size'], config['compression']))
  # Loop over train loader
  for i,(x, y) in enumerate(tqdm(dataloader, file=myargs.stderr)):
    # Stick X into the range [0, 255] since it's coming from the train loader
    x = (255 * ((x + 1) / 2.0)).byte().numpy()
    # Numpyify y
    y = y.numpy()
    # If we're on the first batch, prepare the hdf5
    if i==0:
      with h5.File(config.saved_hdf5, 'w') as f:
        print('Producing dataset of len %d' % len(dataloader.dataset))
        imgs_dset = f.create_dataset(
          'imgs', x.shape,dtype='uint8',
          maxshape=(len(dataloader.dataset), 3,
                    config['image_size'], config['image_size']),
          chunks=(config['chunk_size'], 3,
                  config['image_size'], config['image_size']),
          compression=config['compression'])
        print('Image chunks chosen as ' + str(imgs_dset.chunks))
        imgs_dset[...] = x
        labels_dset = f.create_dataset(
          'labels', y.shape, dtype='int64',
          maxshape=(len(dataloader.dataset),),
          chunks=(config['chunk_size'],),
          compression=config['compression'])
        print('Label chunks chosen as ' + str(labels_dset.chunks))
        labels_dset[...] = y
    # Else append to the hdf5
    else:
      with h5.File(config.saved_hdf5, 'a') as f:
        f['imgs'].resize(f['imgs'].shape[0] + x.shape[0], axis=0)
        f['imgs'][-x.shape[0]:] = x
        f['labels'].resize(f['labels'].shape[0] + y.shape[0], axis=0)
        f['labels'][-y.shape[0]:] = y


class Dataset_HDF5(torch.utils.data.Dataset):
  def __init__(self, root, transform=None, target_transform=None,
               load_in_mem=False, **kwargs):

    self.root = os.path.expanduser(root)
    labels = h5.File(root, 'r')['labels']
    self.num_imgs = len(labels)
    self.classes = list(sorted(set(labels)))

    self.transform = transform
    self.target_transform = target_transform

    # load the entire dataset into memory?
    self.load_in_mem = load_in_mem

    # If loading into memory, do so now
    if self.load_in_mem:
      print('Loading %s into memory...' % root)
      with h5.File(root, 'r') as f:
        self.data = f['imgs'][:]
        self.labels = f['labels'][:]

  def __getitem__(self, index):
    # If loaded the entire dataset in RAM, get image from memory
    if self.load_in_mem:
      img = self.data[index]
      target = self.labels[index]

    # Else load it from disk
    else:
      with h5.File(self.root, 'r') as f:
        img = f['imgs'][index]
        target = f['labels'][index]

    # if self.transform is not None:
    if self.transform:
      img = self.transform(img)
    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, int(target)

  def __len__(self):
    return self.num_imgs


def run(args, myargs):
  myargs.config = getattr(myargs.config, args.command)
  config = myargs.config
  print(pformat(OrderedDict(config)))

  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
  ])
  config.train_dir = os.path.expanduser(config.train_dir)
  dataset = datasets.ImageFolder(config.train_dir, transform)
  nlabels = len(dataset.classes)
  train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=config.batch_size,
    num_workers=config.num_workers,
    shuffle=False, sampler=None
  )
  make_hdf5(train_loader, config.hdf5, myargs)


class TestingUnit(unittest.TestCase):
  def test_Dataset_HDF5(self):
    root = '~/.keras/celeba1024/celebaHQ1024.hdf5'
    root = os.path.expanduser(root)

    transform = transforms.Compose([
      transforms.Lambda(lambda x: x.transpose(1, 2, 0)),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = Dataset_HDF5(root, transform)
    dataset[-1]
    pass