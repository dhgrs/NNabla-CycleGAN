import nnabla as nn
from nnabla import functions as F


def softplus(x):
    return F.log(1 + F.exp(x))


class Updater(object):
    def __init__(self, batch_size, lmd, input_shape, iterator_A, iterator_B,
                 gen_AB, gen_BA, dis_A, dis_B,
                 solver_gen_AB, solver_gen_BA, solver_dis_A, solver_dis_B):
        self.batch_size = batch_size
        self.lmd = lmd
        self.input_shape = input_shape
        self.iterator_A = iterator_A
        self.iterator_B = iterator_B

        self.gen_AB = gen_AB
        self.gen_BA = gen_BA
        self.dis_A = dis_A
        self.dis_B = dis_B

        self.solver_gen_AB = solver_gen_AB
        self.solver_gen_BA = solver_gen_BA
        self.solver_dis_A = solver_dis_A
        self.solver_dis_B = solver_dis_B

        self.make_graph()

        self.gen_AB.set_solver(self.solver_gen_AB)
        self.gen_BA.set_solver(self.solver_gen_BA)
        self.dis_A.set_solver(self.solver_dis_A)
        self.dis_B.set_solver(self.solver_dis_B)

    def make_graph(self):
        # convert A to B
        self.x_A = nn.Variable([self.batch_size] + self.input_shape)
        self.x_AB = self.gen_AB(self.x_A)
        self.x_ABA = self.gen_BA(self.x_AB)

        # convert B to A
        self.x_B = nn.Variable([self.batch_size] + self.input_shape)
        self.x_BA = self.gen_BA(self.x_B)
        self.x_BAB = self.gen_AB(self.x_BA)

        # discriminate A
        self.y_A = self.dis_A(self.x_A)
        self.y_BA = self.dis_A(self.x_BA)

        # discriminate B
        self.y_B = self.dis_B(self.x_B)
        self.y_AB = self.dis_B(self.x_AB)

        # culcurate cycle loss
        self.loss_cyc_A = F.mean(F.abs(self.x_A - self.x_ABA))
        self.loss_cyc_B = F.mean(F.abs(self.x_B - self.x_BAB))
        self.loss_cyc = self.loss_cyc_A + self.loss_cyc_B

        # culcurate GAN loss
        self.loss_gan_gen_A = F.mean(softplus(-self.y_BA))
        self.loss_gan_dis_A = F.mean(
            softplus(-self.y_A) + softplus(self.y_BA))

        self.loss_gan_gen_B = F.mean(softplus(-self.y_AB))
        self.loss_gan_dis_B = F.mean(
            softplus(-self.y_B) + softplus(self.y_AB))

        self.loss_gan_gen = self.loss_gan_gen_A + self.loss_gan_gen_B
        self.loss_gan_dis = self.loss_gan_dis_A + self.loss_gan_dis_B

        # culculate sum of loss
        self.loss_gen = self.lmd * self.loss_cyc + self.loss_gan_gen
        self.loss_dis = self.loss_gan_dis

    def update_gen(self, loss):
        loss.forward()
        self.solver_gen_AB.zero_grad()
        self.solver_gen_BA.zero_grad()
        loss.backward()
        self.solver_gen_AB.update()
        self.solver_gen_BA.update()

    def update_dis(self, loss):
        loss.forward()
        self.solver_dis_A.zero_grad()
        self.solver_dis_B.zero_grad()
        loss.backward()
        self.solver_dis_A.update()
        self.solver_dis_B.update()

    def update(self, iteration):
        # iterate data
        image_A, = self.iterator_A.next()
        image_B, = self.iterator_B.next()

        self.x_A.d = image_A
        self.x_B.d = image_B

        # update
        self.update_gen(self.loss_gen)
        self.update_dis(self.loss_dis)
        # if iteration % 2 == 0:
        #     self.update_gen(self.loss_gan_gen)
        # else:
        #     self.update_dis(self.loss_gan_dis)

        return (self.x_A, self.x_AB, self.x_ABA,
                self.x_B, self.x_BA, self.x_BAB,
                self.loss_cyc, self.loss_gan_gen, self.loss_dis)
