import numpy as np

import nnabla as nn
from nnabla import functions as F
from nnabla import parametric_functions as PF


def bn(x, test):
    return PF.batch_normalization(x, batch_stat=not test)


def upsample(x, c):
    return PF.deconvolution(x, c, kernel=(4, 4), pad=(1, 1), stride=(2, 2),
                            with_bias=False)


def downsample(x, c):
    return PF.convolution(x, c, kernel=(3, 3), pad=(1, 1), stride=(2, 2),
                          with_bias=False)


class Generator(object):
    def __init__(self, scope, hidden_channel, out_channel):
        self.scope = scope
        self.hidden_channel = hidden_channel
        self.out_channel = out_channel

    def set_solver(self, solver):
        with nn.parameter_scope(self.scope):
            solver.set_parameters(nn.get_parameters())

    def __call__(self, x, test=False):
        with nn.parameter_scope(self.scope):
            with nn.parameter_scope('conv1'):
                h1 = F.elu(bn(downsample(x, self.hidden_channel), test))
            with nn.parameter_scope('conv2'):
                h2 = F.elu(bn(downsample(h1, self.hidden_channel // 8), test))
            with nn.parameter_scope('conv3'):
                h3 = F.elu(bn(downsample(h2, self.hidden_channel // 4), test))
            with nn.parameter_scope('conv4'):
                h4 = F.elu(bn(downsample(h3, self.hidden_channel // 2), test))
            with nn.parameter_scope('deconv1'):
                h5 = F.elu(bn(upsample(h4, self.hidden_channel), test))
            with nn.parameter_scope('deconv2'):
                h6 = F.elu(bn(upsample(h5, self.hidden_channel // 2), test))
            with nn.parameter_scope('deconv3'):
                h7 = F.elu(bn(upsample(h6, self.hidden_channel // 4), test))
            with nn.parameter_scope('deconv4'):
                h8 = F.elu(bn(upsample(h7, self.hidden_channel // 8), test))
            with nn.parameter_scope('conv5'):
                y = F.tanh(PF.convolution(h8, self.out_channel,
                                          kernel=(3, 3), pad=(1, 1)))
        return y


class Discriminator(object):
    def __init__(self, scope, hidden_channel):
        self.scope = scope
        self.hidden_channel = hidden_channel

    def set_solver(self, solver):
        with nn.parameter_scope(self.scope):
            solver.set_parameters(nn.get_parameters())

    def __call__(self, x, test=False):
        with nn.parameter_scope(self.scope):
            with nn.parameter_scope('conv1'):
                h1 = F.elu(bn(downsample(x, self.hidden_channel), test))
            with nn.parameter_scope('conv2'):
                h2 = F.elu(bn(downsample(h1, self.hidden_channel // 2), test))
            with nn.parameter_scope('conv3'):
                h3 = F.elu(bn(downsample(h2, self.hidden_channel // 4), test))
            with nn.parameter_scope('conv4'):
                h4 = F.elu(bn(downsample(h3, self.hidden_channel // 8), test))
            with nn.parameter_scope('fc1'):
                f = PF.affine(h4, 1)
        return f
