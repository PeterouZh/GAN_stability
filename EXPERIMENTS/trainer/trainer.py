import functools
import torch
from torch.nn import functional as F
import torch.utils.data
import torch.utils.data.distributed
from torch import autograd

from template_lib.gans import inception_utils


class Trainer(object):
  def __init__(self, generator, discriminator, g_optimizer, d_optimizer,
               gan_type, reg_type, reg_param, args, myargs):
    self.generator = generator
    self.discriminator = discriminator
    self.g_optimizer = g_optimizer
    self.d_optimizer = d_optimizer

    self.gan_type = gan_type
    self.reg_type = reg_type
    self.reg_param = reg_param

    self.args = args
    self.myargs = myargs
    self.config = myargs.config

    # load inception network
    self.inception_metrics = self.inception_metrics_func_create()

  def generator_trainstep(self, y, z):
    assert (y.size(0) == z.size(0))
    toggle_grad(self.generator, True)
    toggle_grad(self.discriminator, False)
    self.generator.train()
    self.discriminator.train()
    self.g_optimizer.zero_grad()

    x_fake = self.generator(z, y)
    d_fake = self.discriminator(x_fake, y)
    gloss_fake = d_fake.mean()
    gloss = -gloss_fake
    gloss.backward()

    self.g_optimizer.step()

    return gloss.item(), gloss_fake.item()

  def discriminator_trainstep(self, x_real, y, z, it=0):
    toggle_grad(self.generator, False)
    toggle_grad(self.discriminator, True)
    self.generator.train()
    self.discriminator.train()
    self.d_optimizer.zero_grad()

    # On real data
    x_real.requires_grad_()

    d_real = self.discriminator(x_real, y)
    dloss_real = d_real.mean()

    # On fake data
    with torch.no_grad():
      x_fake = self.generator(z, y)

    x_fake.requires_grad_()
    d_fake = self.discriminator(x_fake, y)
    dloss_fake = d_fake.mean()

    if self.reg_type == 'real' or self.reg_type == 'real_fake':
      reg = compute_grad2(
        d_real, x_real, lambda_gp=self.reg_param, backward=True)
    elif self.reg_type == 'fake' or self.reg_type == 'real_fake':
      assert 0
      reg = self.reg_param * compute_grad2(
        d_fake, x_fake, lambda_gp=self.reg_param, backward=True)
    elif self.reg_type == 'wgangp':
      reg = self.wgan_gp_reg(
        x_real, x_fake, y, lambda_gp=self.reg_param, backward=True)
    elif self.reg_type == 'wgangp0':
      assert 0
      reg = self.reg_param * self.wgan_gp_reg(x_real, x_fake, y, center=0.)
      reg.backward()

    wd = dloss_real - dloss_fake

    if self.args.command in ['celebaHQ1024_wbgan_gpreal',
                             'celebaHQ1024_wbgan_gp']:
      dloss = -wd + torch.relu(wd - float(self.config.training.bound))
      self.myargs.writer.add_scalar(
        'losses/bound', self.config.training.bound, it)
    else:
      dloss = -wd
    dloss.backward()
    self.d_optimizer.step()

    toggle_grad(self.discriminator, False)

    if self.reg_type == 'none':
      reg = torch.tensor(0.)

    return (dloss.item(), reg.item(), wd.item(),
            dloss_real.item(), dloss_fake.item())

  def compute_loss(self, d_out, target):
    targets = d_out.new_full(size=d_out.size(), fill_value=target)

    if self.gan_type == 'standard':
      loss = F.binary_cross_entropy_with_logits(d_out, targets)
    elif self.gan_type == 'wgan':
      loss = (2 * target - 1) * d_out.mean()
    else:
      raise NotImplementedError

    return loss

  def wgan_gp_reg(self, x_real, x_fake, y, center=1.,
                  lambda_gp=10., backward=False):
    batch_size = y.size(0)
    eps = torch.rand(batch_size, device=y.device).view(batch_size, 1, 1, 1)
    x_interp = (1 - eps) * x_real + eps * x_fake
    x_interp = x_interp.detach()
    x_interp.requires_grad_()
    d_out = self.discriminator(x_interp, y)

    grad_dout = autograd.grad(
      outputs=d_out.sum(), inputs=x_interp,
      create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    grad_norm = grad_dout2.view(batch_size, -1).sum(1)

    reg = (grad_norm.sqrt() - center).pow(2).mean()

    if backward:
      reg = lambda_gp * reg
      reg.backward()
    return reg

  def inception_metrics_func_create(self):
    config = self.myargs.config.fid_metric
    print('Load inception moments: %s' % config.saved_inception_moments)
    inception_metrics = inception_utils.InceptionMetrics(
      saved_inception_moments=config.saved_inception_moments)

    inception_metrics = functools.partial(
      inception_metrics,
      num_inception_images=config.num_inception_images,
      num_splits=10, prints=True)

    return inception_metrics

  def test_FID(self, epoch, zdist, ydist):
    batch_size = self.myargs.config.fid_metric.batch_size
    sample = functools.partial(
      sample_func, G=self.generator, zdist=zdist, ydist=ydist,
      batch_size=batch_size)
    IS_mean, IS_std, FID = self.inception_metrics(
      G=self.generator, z=zdist, show_process=False, use_torch=False,
      sample_func=sample)
    print('IS_mean: %f +- %f, FID: %f' % (IS_mean, IS_std, FID))
    summary = {'IS_mean': IS_mean, 'IS_std': IS_std, 'FID': FID}
    for key in summary:
      self.myargs.writer.add_scalar('test/' + key, summary[key], epoch)


def sample_func(G, zdist, ydist, batch_size):
  with torch.no_grad():
    G.eval()
    ztest = zdist.sample((batch_size,))
    ytest = ydist.sample((batch_size,))

    samples = G(ztest, ytest)

    G.train()
    return samples


# Utility functions
def toggle_grad(model, requires_grad):
  for p in model.parameters():
    p.requires_grad_(requires_grad)


def compute_grad2(d_out, x_in, lambda_gp=10., backward=False):
  batch_size = x_in.size(0)
  grad_dout = autograd.grad(
    outputs=d_out.sum(), inputs=x_in,
    create_graph=True, retain_graph=True, only_inputs=True
  )[0]
  grad_dout2 = grad_dout.pow(2)
  assert (grad_dout2.size() == x_in.size())
  reg = grad_dout2.view(batch_size, -1).sum(1)

  if backward:
    reg = lambda_gp * reg.mean()
    reg.backward(retain_graph=True)
  return reg


def update_average(model_tgt, model_src, beta):
  toggle_grad(model_src, False)
  toggle_grad(model_tgt, False)

  param_dict_src = dict(model_src.named_parameters())

  for p_name, p_tgt in model_tgt.named_parameters():
    p_src = param_dict_src[p_name]
    assert (p_src is not p_tgt)
    p_tgt.copy_(beta * p_tgt + (1. - beta) * p_src)
