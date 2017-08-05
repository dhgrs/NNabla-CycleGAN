import os
import glob
import argparse

import numpy as np

import nnabla as nn
from nnabla.utils.data_iterator import data_iterator
from nnabla import logger as logger
from nnabla import functions as F
from nnabla import parametric_functions as PF
from nnabla import solvers as S
from nnabla import monitor as M

from nets import Generator
from nets import Discriminator
from updater import Updater
from data_source import DataSource
import opt

# Get context.
from nnabla.contrib.context import extension_context
# use CPU or GPU
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU1 ID (negative value indicates CPU)')
args = parser.parse_args()

if args.gpu >= 0:
    extension_module = 'cuda.cudnn'
    device_id = args.gpu
else:
    extension_module = 'cpu'
    device_id = 0

logger.info('Running in %s' % extension_module)
ctx = extension_context(extension_module, device_id=device_id)
nn.set_default_context(ctx)

# define iterator
paths = glob.glob(os.path.join(opt.root_path, '*.png'))
iterator = data_iterator(DataSource(paths, shuffle=True), opt.batch_size)

# define networks
gen = Generator('gen', opt.hidden_shape, opt.hidden_channel, opt.out_channel)
dis = Discriminator('dis', opt.hidden_channel)

# define solvers
solver_gen = S.Adam(opt.learning_rate, beta1=0.5)
solver_dis = S.Adam(opt.learning_rate, beta1=0.5)

# define updater
updater = Updater(opt.batch_size, opt.hidden_shape, opt.input_shape,
                  iterator, gen, dis, solver_gen, solver_dis)

# define monitor
monitor = M.Monitor(opt.monitor_path)
monitor_loss_gen = M.MonitorSeries('Generator loss', monitor, interval=10)
monitor_loss_dis = M.MonitorSeries(
    'Discriminator loss', monitor, interval=10)
monitor_time = M.MonitorTimeElapsed('Time', monitor, interval=100)
monitor_fake = M.MonitorImageTile(
    'Fake images', monitor, normalize_method=lambda x: x + 1 / 2.)

for i in range(opt.max_iter):
    real, fake, loss_gen, loss_dis = updater.update(i)

    monitor_fake.add(i, fake)

    monitor_loss_gen.add(i, loss_gen.d.copy())
    monitor_loss_dis.add(i, loss_dis.d.copy())

    monitor_time.add(i)
