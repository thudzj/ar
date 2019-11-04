import math
import torch

class GradStats:

    def __init__(self, w, momentum=0.9, beta1=0.9, beta2=0.999, beta2_rmsprop=0.9, schedule_decay=4e-3):
        self.momentum = momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta2_rmsprop = beta2_rmsprop
        self.schedule_decay = schedule_decay

        self.step = 0
        self.m_schedule = 1

        self.v = torch.zeros_like(w)
        #self.g_div_h_v = torch.zeros_like(w)
        self.square_grad_sum = torch.zeros_like(w)
        self.first_moment = torch.zeros_like(w)
        self.second_moment = torch.zeros_like(w)
        self.second_moment_rmsprop = torch.zeros_like(w)

    def detach(self,):
        self.v.detach_()
        #self.g_div_h_v.detach_()
        self.square_grad_sum.detach_()
        self.first_moment.detach_()
        self.second_moment.detach_()
        self.second_moment_rmsprop.detach_()

    def update(self, g):
        self.step += 1
        self.momentum_cache_t = self.beta1 * \
            (1. - 0.5 * (0.96 ** (self.step * self.schedule_decay)))
        self.momentum_cache_t_1 = self.beta1 * \
            (1. - 0.5 * (0.96 ** ((self.step + 1) * self.schedule_decay)))
        self.m_schedule_new = self.m_schedule * self.momentum_cache_t
        self.m_schedule_next = self.m_schedule * self.momentum_cache_t * self.momentum_cache_t_1
        self.m_schedule = self.m_schedule_new

        self.v = self.v * self.momentum + g
        self.square_grad_sum = self.square_grad_sum.addcmul(1., g, g)
        self.first_moment = self.first_moment.mul(self.beta1).add(1 - self.beta1, g)
        self.second_moment = self.second_moment.mul(self.beta2).addcmul(1. - self.beta2, g, g)
        self.second_moment_rmsprop = self.second_moment_rmsprop.mul(self.beta2_rmsprop).addcmul(1. - self.beta2_rmsprop, g, g)
        #self.g_div_h_v = (self.g_div_h_v * self.momentum).addcdiv(g, self.second_moment_rmsprop.sqrt().add(1e-8))

def sgd(lr, w, g, grad_stats):
    return w - lr * g

def sgd_momentum(lr, w, g, grad_stats):
    return w - lr * grad_stats.v

def sgd_momentum_nesterov(lr, w, g, grad_stats):
    return w - lr * (g + grad_stats.v * grad_stats.momentum)

def adagrad(lr, w, g, grad_stats, epsilon = 1e-10):
    return w - lr * g / (torch.sqrt(grad_stats.square_grad_sum) + epsilon)

def rmsprop(lr, w, g, grad_stats, epsilon = 1e-8):
    return w - lr * g / (torch.sqrt(grad_stats.second_moment_rmsprop) + epsilon)

def rmsprop_momentum(lr, w, g, grad_stats, epsilon = 1e-8):
    return w - lr * grad_stats.g_div_h_v

def adam(lr, w, g, grad_stats, epsilon = 1e-8):

    second_moment_sqrt_unbiased = (grad_stats.second_moment.sqrt() / math.sqrt(1. - grad_stats.beta2**grad_stats.step)) + epsilon

    bias_correction1 = 1. - grad_stats.beta1**grad_stats.step
    return w - lr / bias_correction1 * grad_stats.first_moment / second_moment_sqrt_unbiased

def nadam(lr, w, g, grad_stats, epsilon = 1e-8):

    second_moment_sqrt_unbiased = (grad_stats.second_moment / (1. - grad_stats.beta2**grad_stats.step)).sqrt().add(epsilon)

    return w.addcdiv(-lr * (1. - grad_stats.momentum_cache_t) / (1. - grad_stats.m_schedule_new), g, second_moment_sqrt_unbiased).addcdiv(-lr * grad_stats.momentum_cache_t_1 / (1. - grad_stats.m_schedule_next), grad_stats.first_moment, second_moment_sqrt_unbiased)

OPTS = [sgd, sgd_momentum, sgd_momentum_nesterov, adagrad, rmsprop, adam, nadam] #rmsprop_momentum,
LRS = [0.05, 0.04, 0.03, 0.02, 0.01]
