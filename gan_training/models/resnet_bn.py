import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
import torch.utils.data.distributed
import numpy as np


class Generator(nn.Module):
    def __init__(self, z_dim, nlabels, size, embed_size=256, nfilter=64, nfilter_max=512, **kwargs):
        super().__init__()
        s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max

        self.z_dim = z_dim

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        self.embedding = nn.Embedding(nlabels, embed_size)
        # self.fc = nn.Linear(z_dim + embed_size, self.nf0*s0*s0)
        self.fc = nn.Sequential(
            nn.Linear(z_dim + embed_size, self.nf0*s0*s0, bias=False),
            nn.BatchNorm1d(self.nf0*s0*s0),
            nn.ReLU())

        blocks = []
        for i in range(nlayers):
            nf0 = min(nf * 2**(nlayers-i), nf_max)
            nf1 = min(nf * 2**(nlayers-i-1), nf_max)
            blocks += [
                ResnetBlock(nf0, nf1, norm='bn'),
                nn.Upsample(scale_factor=2)
            ]

        blocks += [
            ResnetBlock(nf, nf, norm='bn'),
        ]

        self.resnet = nn.Sequential(*blocks)
        self.conv_img = nn.Conv2d(nf, 3, 3, padding=1)

    def forward(self, z, y):
        assert(z.size(0) == y.size(0))
        batch_size = z.size(0)

        if y.dtype is torch.int64:
            yembed = self.embedding(y)
        else:
            yembed = y

        yembed = yembed / torch.norm(yembed, p=2, dim=1, keepdim=True)

        yz = torch.cat([z, yembed], dim=1)
        out = self.fc(yz)
        out = out.view(batch_size, self.nf0, self.s0, self.s0)

        out = self.resnet(out)

        out = self.conv_img(out)
        out = torch.tanh(out)

        return out


class Discriminator(nn.Module):
    def __init__(self, z_dim, nlabels, size, embed_size=256, nfilter=64, nfilter_max=1024):
        super().__init__()
        self.embed_size = embed_size
        s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        blocks = [
            ResnetBlock(nf, nf, norm='in')
        ]

        for i in range(nlayers):
            nf0 = min(nf * 2**i, nf_max)
            nf1 = min(nf * 2**(i+1), nf_max)
            blocks += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlock(nf0, nf1, norm='in'),
            ]

        # self.conv_img = nn.Conv2d(3, 1*nf, 3, padding=1)
        self.conv_img = nn.Sequential(
            nn.Conv2d(3, 1*nf, 3, 1, 1),
            nn.LeakyReLU(0.2))
        self.resnet = nn.Sequential(*blocks)
        self.fc = nn.Linear(self.nf0*s0*s0, nlabels)

    def forward(self, x, y):
        assert(x.size(0) == y.size(0))
        batch_size = x.size(0)

        out = self.conv_img(x)
        out = self.resnet(out)
        out = out.view(batch_size, self.nf0*self.s0*self.s0)
        out = self.fc(out)

        index = Variable(torch.LongTensor(range(out.size(0))))
        if y.is_cuda:
            index = index.cuda()
        out = out[index, y]

        return out


class ResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True, norm=None):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        if norm == 'bn':
            conv_bn_relu = self.conv_bn
        elif norm == 'in':
            conv_bn_relu = self.conv_in
        else:
            conv_bn_relu = nn.Conv2d

        # Submodules
        # self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.conv_0 = conv_bn_relu(self.fin, self.fhidden, 3, 1, 1)
        # self.conv_1 = nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1,
        #                         bias=is_bias)
        self.conv_1 = conv_bn_relu(self.fhidden, self.fout, 3, 1, 1, bias=is_bias)
        if self.learned_shortcut:
            # self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0,
            #                         bias=False)
            self.conv_s = conv_bn_relu(self.fin, self.fout, 1, 1, 0, bias=False)

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = actvn(self.conv_0(x))
        dx = actvn(self.conv_1(dx))
        out = x_s + dx
        # out = x_s + 0.1 * dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = actvn(self.conv_s(x))
        else:
            x_s = x
        return x_s

    @staticmethod
    def conv_bn(in_dim, out_dim, kernel_size, stride, pidding, bias=True):
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size, stride, pidding, bias=bias),
            nn.BatchNorm2d(out_dim))

    @staticmethod
    def conv_in(in_dim, out_dim, kernel_size, stride, pidding, bias=True):
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size, stride, pidding, bias=bias),
            nn.InstanceNorm2d(out_dim, affine=True))


def actvn(x):
    out = F.leaky_relu(x, 2e-1)
    return out
