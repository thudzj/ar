import torch

class GradStats:

    def __init__(self,):
        self.momentum = None
        self.sqr = None

    def update(self, g):
        self.momentum = self.momentum
        self.sqr = self.sqr

def sgd(lr, w, g, grad_stats):
    return w - lr * g

OPTS = [sgd, momentum, adam, rmsprop]
LRS = [0.5, 0.3, 0.1, 0.03]
