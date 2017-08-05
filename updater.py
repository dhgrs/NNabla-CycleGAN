import nnabla as nn
from nnabla import functions as F


class Updater(object):
    def __init__(self, batch_size, hidden_shape, input_shape, iterator,
                 generator, discriminator, solver_gen, solver_dis):
        self.batch_size = batch_size
        self.hidden_shape = hidden_shape
        self.input_shape = input_shape
        self.iterator = iterator
        self.generator = generator
        self.discriminator = discriminator
        self.solver_gen = solver_gen
        self.solver_dis = solver_dis
        self.make_graph()
        self.generator.set_solver(self.solver_gen)
        self.discriminator.set_solver(self.solver_dis)

    def make_graph(self):
        # Fake path
        self.z = nn.Variable([self.batch_size] + self.hidden_shape)
        self.fake = self.generator(self.z)
        self.fake.persistent = True  # Not to clear at backward
        self.pred_fake = self.discriminator(self.fake)
        self.loss_gen = F.mean(F.sigmoid_cross_entropy(
            self.pred_fake, F.constant(1, self.pred_fake.shape)))
        self.fake_dis = self.fake.unlinked()
        self.pred_fake_dis = self.discriminator(self.fake_dis)
        self.loss_dis = F.mean(F.sigmoid_cross_entropy(
            self.pred_fake_dis, F.constant(0, self.pred_fake_dis.shape)))

        # Real path
        self.x = nn.Variable([self.batch_size] + self.input_shape)
        self.pred_real = self.discriminator(self.x)
        self.loss_dis += F.mean(F.sigmoid_cross_entropy(
            self.pred_real, F.constant(1, self.pred_real.shape)))

    def update(self, iteration):
        # iterate data
        image, = self.iterator.next()
        self.x.d = image
        self.z.d = self.generator.make_z(self.batch_size)

        # update
        if iteration % 2 == 0:
            self.loss_gen.forward(clear_no_need_grad=True)
            self.solver_gen.zero_grad()
            self.loss_gen.backward(clear_buffer=True)
            self.solver_gen.update()
        else:
            self.loss_dis.forward(clear_no_need_grad=True)
            self.solver_dis.zero_grad()
            self.loss_dis.backward(clear_buffer=True)
            self.solver_dis.update()

        return self.x, self.fake, self.loss_gen, self.loss_dis
