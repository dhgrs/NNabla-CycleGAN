import os
import glob
import argparse
import csv

import numpy as np

import nnabla as nn
from nnabla.utils.data_iterator import data_iterator
from nnabla.contrib.context import extension_context
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

# use CPU or GPU
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU ID (negative value indicates CPU)')
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
# load data
list_A = []
list_B = []
with open(os.path.join(
        opt.root, 'Anno/list_attr_celeba.txt'), 'r') as f:
    reader = csv.reader(f, delimiter=' ', skipinitialspace=True)
    number = next(reader)
    header = next(reader)
    for row in reader:
        path = row[0].replace('jpg', 'png')
        if row[21] == '1':
            list_A.append(path)
        elif row[21] == '-1':
            list_B.append(path)

iterator_A = data_iterator(
    DataSource(list_A, os.path.join(opt.root, 'Img/img_align_celeba_png/'),
               shuffle=True), opt.batch_size)
iterator_B = data_iterator(
    DataSource(list_B, os.path.join(opt.root, 'Img/img_align_celeba_png/'),
               shuffle=True), opt.batch_size)

# define networks
gen_AB = Generator('gen_AB', opt.hidden_channel, opt.out_channel)
gen_BA = Generator('gen_BA', opt.hidden_channel, opt.out_channel)

dis_A = Discriminator('dis_A', opt.hidden_channel)
dis_B = Discriminator('dis_B', opt.hidden_channel)

# define solvers
solver_gen_AB = S.Adam(opt.learning_rate, beta1=0.5)
solver_gen_BA = S.Adam(opt.learning_rate, beta1=0.5)

solver_dis_A = S.Adam(opt.learning_rate, beta1=0.5)
solver_dis_B = S.Adam(opt.learning_rate, beta1=0.5)

# define updater
updater = Updater(opt.batch_size, opt.input_shape, iterator_A, iterator_B,
                  gen_AB, gen_BA, dis_A, dis_B,
                  solver_gen_AB, solver_gen_BA, solver_dis_A, solver_dis_B)

# define monitor
monitor = M.Monitor(opt.monitor_path)
monitor_loss_cyc = M.MonitorSeries('Cycle loss', monitor, interval=opt.monitor_interval)
monitor_loss_gen = M.MonitorSeries('Generator loss', monitor, interval=opt.monitor_interval)

monitor_loss_dis = M.MonitorSeries(
    'Discriminator loss', monitor, interval=opt.monitor_interval)
monitor_time = M.MonitorTimeElapsed('Time', monitor, interval=opt.monitor_interval)
monitor_A = M.MonitorImageTile(
    'Fake images_A', monitor, normalize_method=lambda x: x + 1 / 2.,
    interval=opt.generate_interval)
monitor_B = M.MonitorImageTile(
    'Fake images_B', monitor, normalize_method=lambda x: x + 1 / 2.,
    interval=opt.generate_interval)

# training loop
for i in range(opt.max_iter):
    (x_A, x_AB, x_ABA, x_B, x_BA, x_BAB,
     loss_cyc, loss_gen, loss_dis) = updater.update(i)

    As = np.concatenate((x_A.d, x_AB.d, x_ABA.d), axis=3)
    Bs = np.concatenate((x_B.d, x_BA.d, x_BAB.d), axis=3)

    monitor_A.add(i, As)
    monitor_B.add(i, Bs)

    monitor_loss_cyc.add(i, loss_cyc.d.copy())
    monitor_loss_gen.add(i, loss_gen.d.copy())
    monitor_loss_dis.add(i, loss_dis.d.copy())

    if (i + 1) % opt.save_interval == 0:
        with nn.parameter_scope('gen_AB'):
            nn.save_parameters(
                os.path.join(opt.model_save_path, 'gen_AB_{}.h5'.format(i)))
        with nn.parameter_scope('gen_BA'):
            nn.save_parameters(
                os.path.join(opt.model_save_path, 'gen_BA_{}.h5'.format(i)))

        with nn.parameter_scope('dis_A'):
            nn.save_parameters(
                os.path.join(opt.model_save_path, 'dis_A_{}.h5'.format(i)))
        with nn.parameter_scope('dis_B'):
            nn.save_parameters(
                os.path.join(opt.model_save_path, 'dis_B_{}.h5'.format(i)))

    monitor_time.add(i)
