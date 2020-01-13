# coding: utf-8

import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl

class QFunction(chainer.Chain):
    def __init__(self, obs_size, n_actions):
        super(QFunction, self).__init__(##python2.x用
        #super().__init__(#python3.x用
            # conv11 = L.Convolution2D(1, 64, ksize=(1, 2)),
            # conv12 = L.Convolution2D(None, 128, ksize=(1, 2)),
            # conv13 = L.Convolution2D(None, 256, ksize=(1, 2)),
            # conv14 = L.Convolution2D(None, 512, ksize=(4, 1)),

            # conv21 = L.Convolution2D(1, 64, ksize=(2, 1)),
            # conv22 = L.Convolution2D(None, 128, ksize=(2, 1)),
            # conv23 = L.Convolution2D(None, 256, ksize=(2, 1)),
            # conv24 = L.Convolution2D(None, 512, ksize=(1, 4)),

            l1=L.Linear(None, 1024),
            l2=L.Linear(None, 512),
            l3=L.Linear(None, 256),
            l4=L.Linear(None, 128),
            l5=L.Linear(None, n_actions))

    def __call__(self, x, test=False):
        """
        x ; 観測#ここの観測って、stateとaction両方？
        test : テストモードかどうかのフラグ
        """

        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = F.relu(self.l3(h))
        h = F.relu(self.l4(h))
        return chainerrl.action_value.DiscreteActionValue(self.l5(h))

        debug_print = False
        if debug_print: print(x.shape)
        h1 = F.relu(self.conv11(x))
        if debug_print: print(h1.shape)
        h1 = F.relu(self.conv12(h1))
        if debug_print: print(h1.shape)
        h1 = F.relu(self.conv13(h1))
        if debug_print: print(h1.shape)
        h1 = F.relu(self.conv14(h1))
        if debug_print: print(h1.shape)

        if debug_print: print(x.shape)
        h2 = F.relu(self.conv21(x))
        if debug_print: print(h2.shape)
        h2 = F.relu(self.conv22(h2))
        if debug_print: print(h2.shape)
        h2 = F.relu(self.conv23(h2))
        if debug_print: print(h2.shape)
        h2 = F.relu(self.conv24(h2))
        if debug_print: print(h2.shape)

        if debug_print: print("PO")

        h = F.concat((h1, h2))
        if debug_print: print(h.shape)
        h = h.reshape((-1, 1024))
        if debug_print: print(h.shape)

        if debug_print: print(h.shape)
        h = F.relu(self.l1(h))
        h = F.relu(self.l2(h))
        h = F.relu(self.l3(h))
        h = F.relu(self.l4(h))

        return chainerrl.action_value.DiscreteActionValue(self.l5(h))
