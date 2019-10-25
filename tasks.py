import os
import numpy

class Task:
    '''
    A bi-level task:
    the inner optimization problem is w_o = argmin_w (w^2 - p*alpha*w + alpha^2), given p!=0,
    the outer optimization problem is min_{alpha} (w_o * alpha - q * alpha + 1).
    The w_o can be analytically calculated as w_o = p*alpha/2.
    Substitute this into the outer optimization, we get: min_{alpha} (p/2 * alpha^2 - q * alpha + 1),
    whose optima is alpha_o = q/p
    '''
    def __init__(self, p=2, q=2):
        self.p = p
        self.q = q
        self.optimal_alpha = q/p
        self.optimal_w = q/2

    # return the reward of the current alpha
    def get_reward(self, alpha):
        return -(self.optimal_alpha - alpha)**2

    # inner loss
    def inner_loss(self, w, alpha):
        return w**2 - self.p*w*alpha + alpha**2

    # calculate the inner gradient for parameter w
    def inner_grad(self, w, alpha):
        return w * 2 - alpha * self.p

    # outer loss
    def outer_loss(self, w, alpha):
        return w * alpha - self.q * alpha + 1

def random_task():
    return Task(numpy.random.uniform(1, 4), numpy.random.uniform(1e-8, 4))
