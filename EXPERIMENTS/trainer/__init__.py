
exe_dict = {
  'celebaHQ1024_wgan_gpreal': 'train',
  'celebaHQ1024_wbgan_gpreal': 'train',
  'celebaHQ1024_wgan_gp': 'train',
  'celebaHQ1024_wbgan_gp': 'train',
  'cifar10_wbgan_gpreal': 'train'
}

from . import trainer

trainer_dict = {
  'celebaHQ1024_wgan_gpreal': trainer.Trainer,
  'celebaHQ1024_wbgan_gpreal': trainer.Trainer,
  'celebaHQ1024_wgan_gp': trainer.Trainer,
  'celebaHQ1024_wbgan_gp': trainer.Trainer,
  'cifar10_wbgan_gpreal': trainer.Trainer
}