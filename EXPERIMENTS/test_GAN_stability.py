import os
import sys
import unittest
import argparse

from template_lib import utils


class Testing_prepare_data(unittest.TestCase):

  def test_celebaHQ1024_make_hdf5(self):
    """
    Usage:
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
        export PORT=6112
        export TIME_STR=1
        export PYTHONPATH=../submodule:..:.
        python -c "import test_GAN_stability; \
        test_GAN_stability.Testing_prepare_data().test_celebaHQ1024_make_hdf5()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6007'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

    # func name
    outdir = os.path.join('results/dataset', sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config configs/gan_stability_celebaHQ.yaml
            --command celebaHQ1024_make_hdf5
            --resume False --resume_path None
            --resume_root None
            """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args.CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str
    args, argv_str = build_args()

    # parse the config json file
    args = utils.config.process_config(outdir=outdir, config_file=args.config,
                                       resume_root=args.resume_root, args=args,
                                       myargs=myargs)
    from TOOLS import make_hdf5
    make_hdf5.run(args, myargs)
    input('End %s' % outdir)
    return

  def test_cifar10_inception_moments(self):
    """
    Usage:
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
        export PORT=6112
        export TIME_STR=1
        export PYTHONPATH=../submodule:..:.
        python -c "import test_GAN_stability; \
        test_GAN_stability.Testing_prepare_data().test_cifar10_inception_moments()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6007'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

    # func name
    outdir = os.path.join('results/dataset', sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config configs/wgan_gp_cifar10.yaml
            --command cifar10_inception_moments
            --resume False --resume_path None
            --resume_root None
            """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args.CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str

    args, argv_str = build_args()

    # parse the config json file
    args = utils.config.process_config(outdir=outdir, config_file=args.config,
                                       resume_root=args.resume_root, args=args,
                                       myargs=myargs)
    from TOOLS import calculate_inception_moments
    calculate_inception_moments.get_cifar10_inception_moments(args, myargs)
    input('End %s' % outdir)
    return


class Testing_GAN_stability(unittest.TestCase):

  def test_train_celebaHQ1024_wgan_gpreal(self):
    """
    Usage:
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
        export PORT=6112
        export TIME_STR=1
        export PYTHONPATH=../submodule:..:.
        python -c "import test_GAN_stability; \
        test_GAN_stability.Testing_GAN_stability().test_train_celebaHQ1024_wgan_gpreal()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6111'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

    # func name
    outdir = os.path.join('results/GAN_stability', sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config configs/gan_stability_celebaHQ.yaml
            --command celebaHQ1024_wgan_gpreal
            --resume False --resume_path None
            --resume_root None
            """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args.CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str
    args, argv_str = build_args()

    # parse the config json file
    args = utils.config.process_config(outdir=outdir, config_file=args.config,
                                       resume_root=args.resume_root, args=args,
                                       myargs=myargs)
    from trainer import run
    run.main(args, myargs)
    input('End %s' % outdir)
    return

  def test_train_celebaHQ1024_wbgan_gpreal(self):
    """
    Usage:
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
        export PORT=6111
        export TIME_STR=1
        export PYTHONPATH=../submodule:..:.
        python -c "import test_GAN_stability; \
        test_GAN_stability.Testing_GAN_stability().test_train_celebaHQ1024_wbgan_gpreal()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6007'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

    # func name
    outdir = os.path.join('results/GAN_stability',
                          sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config configs/gan_stability_celebaHQ.yaml
            --command celebaHQ1024_wbgan_gpreal
            --resume False --resume_path None
            --resume_root None
            """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args.CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str

    args, argv_str = build_args()

    # parse the config json file
    args = utils.config.process_config(outdir=outdir, config_file=args.config,
                                       resume_root=args.resume_root, args=args,
                                       myargs=myargs)
    from trainer import run
    run.main(args, myargs)
    input('End %s' % outdir)
    return

  def test_train_celebaHQ1024_wgan_gp(self):
    """
    Usage:
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
        export PORT=6006
        export TIME_STR=1
        export PYTHONPATH=../submodule:..:.
        python -c "import test_GAN_stability; \
        test_GAN_stability.Testing_GAN_stability().test_train_celebaHQ1024_wgan_gp()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6007'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

    # func name
    outdir = os.path.join('results/GAN_stability',
                          sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config configs/gan_stability_celebaHQ.yaml
            --command celebaHQ1024_wgan_gp
            --resume False --resume_path None
            --resume_root None
            """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args.CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str

    args, argv_str = build_args()

    # parse the config json file
    args = utils.config.process_config(outdir=outdir, config_file=args.config,
                                       resume_root=args.resume_root, args=args,
                                       myargs=myargs)
    from trainer import run
    run.main(args, myargs)
    input('End %s' % outdir)
    return

  def test_train_celebaHQ1024_wbgan_gp(self):
    """
    Usage:
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
        export PORT=6006
        export TIME_STR=1
        export PYTHONPATH=../submodule:..:.
        python -c "import test_GAN_stability; \
        test_GAN_stability.Testing_GAN_stability().test_train_celebaHQ1024_wbgan_gp()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6007'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

    # func name
    outdir = os.path.join('results/GAN_stability',
                          sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config configs/gan_stability_celebaHQ.yaml
            --command celebaHQ1024_wbgan_gp
            --resume False --resume_path None
            --resume_root None
            """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args.CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str

    args, argv_str = build_args()

    # parse the config json file
    args = utils.config.process_config(outdir=outdir, config_file=args.config,
                                       resume_root=args.resume_root, args=args,
                                       myargs=myargs)
    from trainer import run
    run.main(args, myargs)
    input('End %s' % outdir)
    return


class Testing_cifar10(unittest.TestCase):

  def test_train_cifar10_wbgan_gpreal(self):
    """
    Usage:
        export CUDA_VISIBLE_DEVICES=5
        export PORT=6111
        export TIME_STR=1
        export PYTHONPATH=../submodule:..:.
        python -c "import test_GAN_stability; \
        test_GAN_stability.Testing_cifar10().test_train_cifar10_wbgan_gpreal()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6007'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

    # func name
    outdir = os.path.join('results/cifar10', sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config configs/wgan_gp_cifar10.yaml
            --command cifar10_wbgan_gpreal
            --resume False --resume_path None
            --resume_root None
            """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args.CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str
    args, argv_str = build_args()

    # parse the config json file
    args = utils.config.process_config(outdir=outdir, config_file=args.config,
                                       resume_root=args.resume_root, args=args,
                                       myargs=myargs)
    from trainer import run
    run.main(args, myargs)
    input('End %s' % outdir)
    return

