import os
from collections import OrderedDict
from pprint import pformat
import torch.nn.functional as F
import numpy as np
import tqdm
import torch
import unittest

from template_lib.gans import inception_utils


def create_inception_moments(data_loader, config, myargs):
  # Load inception net
  net = inception_utils.load_inception_net(parallel=config.parallel)
  pool, logits, labels = [], [], []
  device = 'cuda'
  for i, (x, y) in enumerate(tqdm.tqdm(data_loader, file=myargs.stdout)):
    x = x.to(device)
    with torch.no_grad():
      pool_val, logits_val = net(x)
      pool += [np.asarray(pool_val.cpu())]
      logits += [np.asarray(F.softmax(logits_val, 1).cpu())]
      labels += [np.asarray(y.cpu())]

  pool, logits, labels = [np.concatenate(item, 0) for item in \
                          [pool, logits, labels]]
  # uncomment to save pool, logits, and labels to disk
  # logger.info('Saving pool, logits, and labels to disk...')
  # np.savez(config['dataset']+'_inception_activations.npz',
  #           {'pool': pool, 'logits': logits, 'labels': labels})
  # Calculate inception metrics and report them
  print('Calculating inception metrics...')
  IS_mean, IS_std = inception_utils.calculate_inception_score(logits)
  print('Training data has IS of %5.5f +/- %5.5f'%(IS_mean, IS_std))
  # Prepare mu and sigma, save to disk. Remove "hdf5" by default
  # (the FID code also knows to strip "hdf5")
  print('Calculating means and covariances...')
  mu, sigma = np.mean(pool, axis=0), np.cov(pool, rowvar=False)
  print('Saving calculated means and covariances to: %s' \
        %config.saved_inception_moments)
  np.savez(os.path.expanduser(config.saved_inception_moments),
           **{'mu': mu, 'sigma': sigma})


def get_cifar10_inception_moments(args, myargs):
  myargs.config = getattr(myargs.config, args.command)
  config = myargs.config
  print(pformat(OrderedDict(config)))

  from gan_training.inputs import get_dataset
  dataset, _ = get_dataset(name=config.name, data_dir=config.train_dir)

  train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=config.batch_size,
    num_workers=config.num_workers,
    shuffle=False, pin_memory=False, sampler=None, drop_last=False
  )
  create_inception_moments(
    data_loader=train_loader, config=config.inception_moments, myargs=myargs)


