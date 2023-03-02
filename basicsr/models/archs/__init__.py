import importlib
from os import path as osp
from torch.nn import init
import torch.nn as nn
import functools
from timm.models.layers import trunc_normal_

from basicsr.utils import scandir

# automatically scan and import arch modules
# scan all the files under the 'archs' folder and collect files ending with
# '_arch.py'
arch_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = [
    osp.splitext(osp.basename(v))[0] for v in scandir(arch_folder)
    if v.endswith('_arch.py')
]
# import all the arch modules
_arch_modules = [
    importlib.import_module(f'basicsr.models.archs.{file_name}')
    for file_name in arch_filenames
]


def dynamic_instantiation(modules, cls_type, opt):
    """Dynamically instantiate class.

    Args:
        modules (list[importlib modules]): List of modules from importlib
            files.
        cls_type (str): Class type.
        opt (dict): Class initialization kwargs.

    Returns:
        class: Instantiated class.
    """

    for module in modules:
        cls_ = getattr(module, cls_type, None)
        if cls_ is not None:
            break
    if cls_ is None:
        raise ValueError(f'{cls_type} is not found.')
    return cls_(**opt)

def define_network(opt):
    network_type = opt.pop('type')
    net = dynamic_instantiation(_arch_modules, network_type, opt)
    return net

def define_g(opt):
    network_type = opt.pop('type')
    net = dynamic_instantiation(_arch_modules, network_type, opt)

    init_weights(net,
                 init_type='trunc_normal',
                 init_bn_type='constant',
                 gain=0.2)

    return net


# --------------------------------------------
# Discriminator, netD, D
# --------------------------------------------
def define_d(opt):

    net_type = opt['type']

    # ----------------------------------------
    # discriminator_vgg_96
    # ----------------------------------------
    if net_type == 'discriminator_vgg_96':
        from discriminator_arch import Discriminator_VGG_96 as discriminator
        netD = discriminator(in_nc=opt['in_nc'],
                             base_nc=opt['base_nc'],
                             ac_type=opt['act_mode'])

    # ----------------------------------------
    # discriminator_vgg_128
    # ----------------------------------------
    elif net_type == 'discriminator_vgg_128':
        from discriminator_arch import Discriminator_VGG_128 as discriminator
        netD = discriminator(in_nc=opt['in_nc'],
                             base_nc=opt['base_nc'],
                             ac_type=opt['act_mode'])

    # ----------------------------------------
    # discriminator_vgg_192
    # ----------------------------------------
    elif net_type == 'discriminator_vgg_192':
        from discriminator_arch import Discriminator_VGG_192 as discriminator
        netD = discriminator(in_nc=opt['in_nc'],
                             base_nc=opt['base_nc'],
                             ac_type=opt['act_mode'])

    # ----------------------------------------
    # discriminator_vgg_128_SN
    # ----------------------------------------
    elif net_type == 'discriminator_vgg_128_SN':
        from discriminator_arch import Discriminator_VGG_128_SN as discriminator
        netD = discriminator()

    elif net_type == 'patchgan_batch':
        from discriminator_arch import Discriminator_PatchGAN as discriminator
        netD = discriminator(input_nc=opt['in_nc'],
                             ndf=opt['base_nc'],
                             n_layers=opt['n_layers'],
                             norm_type='batch')

    elif net_type == 'patchgan_instance':
        from discriminator_arch import Discriminator_PatchGAN as discriminator
        netD = discriminator(input_nc=opt['in_nc'],
                             ndf=opt['base_nc'],
                             n_layers=opt['n_layers'],
                             norm_type='instance')

    elif net_type == 'patchgan_spectral':
        from discriminator_arch import Discriminator_PatchGAN as discriminator
        netD = discriminator(input_nc=opt['in_nc'],
                             ndf=opt['base_nc'],
                             n_layers=opt['n_layers'],
                             norm_type='spectral')

    elif net_type == 'UNetDiscriminatorSN':
        from discriminator_arch import UNetDiscriminatorSN as discriminator
        netD = discriminator(in_nc=opt['in_nc'],
                             base_nc=opt['base_nc'],
                             skip_connection=True)

    else:
        raise NotImplementedError('netD [{:s}] is not found.'.format(net_type))

    # ----------------------------------------
    # initialize weights
    # ----------------------------------------
    init_weights(netD,
                 init_type=opt['init_type'],
                 init_bn_type=opt['init_bn_type'],
                 gain=opt['init_gain'])

    return netD


"""
# --------------------------------------------
# weights initialization
# --------------------------------------------
"""


def init_weights(net, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
    """
    # Kai Zhang, https://github.com/cszn/KAIR
    #
    # Args:
    #   init_type:
    #       normal; normal; xavier_normal; xavier_uniform;
    #       kaiming_normal; kaiming_uniform; orthogonal
    #   init_bn_type:
    #       uniform; constant
    #   gain:
    #       0.2
    """
    print('Initialization method [{:s} + {:s}], gain is [{:.2f}]'.format(init_type, init_bn_type, gain))

    def init_fn(m, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
        classname = m.__class__.__name__

        if classname.find('Conv') != -1 or classname.find('Linear') != -1:

            if init_type == 'normal':
                init.normal_(m.weight.data, 0, 0.1)
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'uniform':
                init.uniform_(m.weight.data, -0.2, 0.2)
                m.weight.data.mul_(gain)

            elif init_type == 'xavier_normal':
                init.xavier_normal_(m.weight.data, gain=gain)
                m.weight.data.clamp_(-1, 1)

            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=gain)

            elif init_type == 'kaiming_normal':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'kaiming_uniform':
                init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.mul_(gain)

            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)

            elif init_type == 'trunc_normal':
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)

            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_type))

            if m.bias is not None:
                m.bias.data.zero_()

        elif classname.find('BatchNorm2d') != -1:

            if init_bn_type == 'uniform':  # preferred
                if m.affine:
                    init.uniform_(m.weight.data, 0.1, 1.0)
                    init.constant_(m.bias.data, 0.0)
            elif init_bn_type == 'constant':
                if m.affine:
                    init.constant_(m.weight.data, 1.0)
                    init.constant_(m.bias.data, 0.0)
            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_bn_type))
        elif classname.find('LayerNorm') != -1:
            if init_bn_type == 'constant':
                if isinstance(m, nn.LayerNorm):
                    init.constant_(m.bias, 0.0)
                    init.constant_(m.weight, 1.0)
            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_bn_type))


    fn = functools.partial(init_fn, init_type=init_type, init_bn_type=init_bn_type, gain=gain)
    net.apply(fn)