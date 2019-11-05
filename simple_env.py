import numpy as np
import torch
from optimizers import OPTS, LRS, GradStats
from tasks import Task, random_task
import matplotlib.pyplot as plt

class SimpleEnv:

    def __init__(self, p=None, q=None, num_tasks_per_batch=None, testing=False, seed=1000):
        # hyper-parameters
        if seed:
            np.random.seed(seed)
            # torch.cuda.set_device(args.gpu)
            # cudnn.benchmark = True
            torch.manual_seed(seed)
            # cudnn.enabled=True
            # torch.cuda.manual_seed(args.seed)

        self.weight_decay_w = 0
        self.beta2_rmsprop = 0.9
        self.outer_T = 40
        self.inner_T = 6
        self.testing = testing
        self.num_tasks_per_batch = 1
        self.n_opts = len(OPTS)

        # initialize
        # self.reset(p, q)

    def step(self, action):
        # action: opt, lr
        opt = action[0]
        lr = action[1]
        episode_over = False
        if opt == self.n_opts:
            self.outer_count = self.outer_count + 1
            self.w = self.ws[-1]

            self.optimizer_alpha.zero_grad()
            self.task.outer_loss(self.w, self.alpha).sum().backward()
            self.optimizer_alpha.step()
            self.lr_scheduler.step()

            self.points_x.append(self.alpha[0].item())
            self.points_y.append(self.w[0].item())
            self.grad_stats.detach()
            self.ws = [self.w.detach_().requires_grad_()]

            self.o_loss = self.task.outer_loss(self.ws[-1], self.alpha)
            self.o_grad = torch.autograd.grad(outputs=self.o_loss, inputs=self.alpha,
                                            grad_outputs=torch.ones_like(self.o_loss))[0]
            if self.outer_count == 1:
                self.o_loss_mv = self.o_loss.data.clone()
                self.o_grad_norm_mv = self.o_grad.data.norm()
            else:
                self.o_loss_mv = self.o_loss_mv*0.9 + self.o_loss.data*0.1
                self.o_grad_norm_mv = self.o_grad_norm_mv*0.9 + self.o_grad.data.norm()*0.1

            self.i_grad_change = -self.i_grad.data
            self.i_loss = self.task.inner_loss(self.ws[-1], self.alpha) + self.weight_decay_w/2. * (self.ws[-1])**2
            self.i_grad = torch.autograd.grad(outputs=self.i_loss, inputs=self.ws[-1],
                                            grad_outputs=torch.ones_like(self.i_loss),
                                            create_graph=True, retain_graph=True, only_inputs=True)[0]
            self.i_grad_change += self.i_grad.data
            self.grad_stats.update(self.i_grad)
            self.i_loss_mv = self.i_loss_mv*0.9 + self.i_loss.data*0.1
            self.i_grad_norm_mv = self.i_grad_norm_mv*0.9 + self.i_grad.data.norm()*0.1

            ob = torch.stack([self.i_loss, self.i_loss-self.i_loss_mv, self.i_grad.norm().view(1), (self.i_grad.norm().add(1e-16).log()-self.i_grad_norm_mv.log()).view(1), self.i_grad_change.norm().view(1), self.o_loss, self.o_loss-self.o_loss_mv, torch.tensor([0.]), torch.tensor([float(self.outer_count)/self.outer_T])], 1).detach()

            if self.inner_count > 0 and self.i_loss.item() <= 100:
                reward = self.task.get_reward(self.alpha, self.ws[-1]).item()
                self.inner_count = 0
            else:
                reward = -1
                episode_over = True
            if self.outer_count == self.outer_T:
                episode_over = True
            self.points.append(reward)
        else:
            self.ws.append(OPTS[opt](LRS[lr], self.ws[-1], self.i_grad, self.grad_stats))
            self.inner_count += 1

            self.i_grad_change = -self.i_grad.data
            self.i_loss = self.task.inner_loss(self.ws[-1], self.alpha) + self.weight_decay_w/2. * (self.ws[-1])**2
            self.i_grad = torch.autograd.grad(outputs=self.i_loss, inputs=self.ws[-1],
                                            grad_outputs=torch.ones_like(self.i_loss),
                                            create_graph=True, retain_graph=True, only_inputs=True)[0]
            self.i_grad_change += self.i_grad.data
            self.grad_stats.update(self.i_grad)
            self.i_loss_mv = self.i_loss_mv*0.9 + self.i_loss.data*0.1
            self.i_grad_norm_mv = self.i_grad_norm_mv*0.9 + self.i_grad.data.norm()*0.1

            ob = torch.stack([self.i_loss, self.i_loss-self.i_loss_mv, self.i_grad.norm().view(1), (self.i_grad.norm().add(1e-16).log()-self.i_grad_norm_mv.log()).view(1), self.i_grad_change.norm().view(1), self.o_loss, self.o_loss-self.o_loss_mv, torch.tensor([float(self.inner_count)/self.inner_T]), torch.tensor([float(self.outer_count)/self.outer_T])], 1).detach()

            if self.inner_count < self.inner_T and self.i_loss.item() <= 100:
                reward = 0
            else:
                reward = -1
                episode_over = True

        return ob, reward, episode_over, {}

    def reset(self, p=None, q=None):
        # initialize a task
        if p and q:
            self.task = Task(p=np.array([p]), q=np.array([q]), num_tasks_per_batch=1)
        else:
            self.task = random_task(self.num_tasks_per_batch)

        # define variables
        self.w = torch.tensor([-2.]*self.num_tasks_per_batch, requires_grad=True)
        self.alpha = torch.tensor([2.]*self.num_tasks_per_batch, requires_grad=True)
        self.ws = [self.w]

        # initialize the statistics of gradients for w
        self.grad_stats = GradStats(self.w, beta2_rmsprop=self.beta2_rmsprop)

        # define optimizer for alpha
        self.optimizer_alpha = torch.optim.SGD([self.alpha], lr=.1)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_alpha, 0.95)

        self.o_loss = torch.zeros_like(self.alpha)
        self.o_grad = torch.zeros_like(self.alpha)

        self.points = []
        self.points_x = [self.alpha[0].item()]
        self.points_y = [self.w[0].item()]

        self.i_loss = self.task.inner_loss(self.ws[-1], self.alpha) + self.weight_decay_w/2. * (self.ws[-1])**2
        self.i_grad = torch.autograd.grad(outputs=self.i_loss, inputs=self.ws[-1],
                                        grad_outputs=torch.ones_like(self.i_loss),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        self.grad_stats.update(self.i_grad)

        self.i_loss_mv = self.i_loss.clone().detach_()
        self.i_grad_norm_mv = self.i_grad.data.norm()
        self.o_loss_mv = self.o_loss.clone().detach_()
        self.o_grad_norm_mv = self.o_grad.data.norm()
        self.i_grad_change = self.i_grad.clone().detach_()

        #whether in inner optimization
        self.inner_count = 0
        self.outer_count = 0

        #print('reset')
        return torch.stack([self.i_loss, self.i_loss-self.i_loss_mv, self.i_grad.norm().view(1), (self.i_grad.norm().log()-self.i_grad_norm_mv.log()).view(1), self.i_grad_change.norm().view(1), self.o_loss, self.o_loss-self.o_loss_mv, torch.tensor([0.]), torch.tensor([0.])], 1).detach()

    def render(self):
        plt.subplot(1,2,1)
        plt.plot(self.points)

        plt.subplot(1,2,2)
        plt.plot(self.points_x, self.points_y)
        tmp = np.max(self.points_x)
        plt.plot([0, tmp+1], [0, (tmp+1)*self.task.p[0]/2.], 'r--')
        plt.plot([self.task.optimal_alpha[0]], [self.task.optimal_w[0]], 'g^')
        plt.show()
        #plt.savefig("reward.pdf")

    def close(self):
        print('close')
