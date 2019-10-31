import numpy as np
import gym
import torch
from gym import error, spaces, utils
from gym.utils import seeding
from gym_auto.envs.optimizers import OPTS, LRS, GradStats
from gym_auto.envs.tasks import Task, random_task
import matplotlib.pyplot as plt

class AutoEnv(gym.Env):
    #actionlist = [op,lr], 7 action for op; 4 action for lr
    #OPTS = [sgd, sgd_momentum, sgd_momentum_nesterov, adagrad, rmsprop, adam, nadam]
    #LRS = [0.5, 0.3, 0.1, 0.03]
    metadata = {'render.modes':['human']}

    def __init__(self):
        # hyper-parameters
        self.weight_decay_w = 0
        self.beta2_rmsprop = 0.9
        self.outer_T = 100
        self.inner_T = 5
        self.num_tasks_per_batch = 10
        # initialize the policy network and the corresponding optimizer
        # policy_net = MLP(20)
        # optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=.01)
        
        # initialize a task
        self.task = random_task(self.num_tasks_per_batch)

        # define variables
        self.w = torch.tensor([-2.]*self.num_tasks_per_batch, requires_grad=True)
        self.alpha = torch.tensor([2.]*self.num_tasks_per_batch, requires_grad=True)
        
        # define optimizer
        self.optimizer_alpha = torch.optim.SGD([self.alpha], lr=.3)
        
        # initialize the statistics of gradients
        self.grad_stats = GradStats(self.w, beta2_rmsprop=self.beta2_rmsprop)
        
        self.o_loss = torch.zeros_like(self.alpha)
        self.o_grad = torch.zeros_like(self.alpha)
        
        self.tmp = torch.ones_like(self.alpha)/self.num_tasks_per_batch
        
        self.grad_stats.detach()
        self.ws = [self.w.detach_().requires_grad_()]
        
        #whether in inner optimization
        self.inner_count = 0
        self.outer_count = 0
        
        self.points = []
        self.points_x = []
        self.points_y = []
        
        self.i_loss = self.task.inner_loss(self.ws[-1], self.alpha) + self.weight_decay_w/2. * (self.ws[-1])**2
        self.i_grad = torch.autograd.grad(outputs=self.i_loss, inputs=self.ws[-1], grad_outputs=torch.ones_like(self.i_loss),
                                          create_graph=True, retain_graph=True, only_inputs=True)[0]
        self.grad_stats.update(self.i_grad)
    
    def step(self, action):
        #action: opt, lr
        opt = action[0]
        lr = action[1]
        self.inner_count = (self.inner_count + 1) % self.inner_T
        self.ws.append(OPTS[opt](LRS[lr], self.ws[-1], self.i_grad, self.grad_stats))
        reward = 0
        episode_over = False
        
        #whether in inner optimization
        if self.inner_count == 0:
            self.outer_count = self.outer_count + 1
            self.w = self.ws[-1]
        
            self.o_loss = self.task.outer_loss(self.w, self.alpha).sum()
            self.optimizer_alpha.zero_grad()
            self.o_loss.backward()
            self.o_grad = self.alpha.grad.clone().detach()
            self.optimizer_alpha.step()
            
            # get reward and update the policy
            reward = self.task.get_reward(self.alpha)
            # optimizer_policy.zero_grad()
            # policy_loss = -logp * reward.mean()
            # policy_loss.backward()
            # optimizer_policy.step()
            
            self.points.append(torch.mean(reward))
            self.points_x.append(self.ws[-1][0].item())
            self.points_y.append(self.alpha[0].item())
            print("Training ite", self.outer_count, reward)
            
        else:
            reward = 0
        
        self.i_loss = self.task.inner_loss(self.ws[-1], self.alpha) + self.weight_decay_w/2. * (self.ws[-1])**2
        self.i_grad = torch.autograd.grad(outputs=self.i_loss, inputs=self.ws[-1], grad_outputs=torch.ones_like(self.i_loss),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        self.grad_stats.update(self.i_grad)
        ob = self.ws[-1], self.i_grad, self.grad_stats
        
        if self.outer_count == self.outer_T:
            episode_over = True
        else:
            episode_over = False
        
        return  ob, reward, episode_over,{}
        
    def reset(self):
        # initialize a task
        self.task = random_task(self.num_tasks_per_batch)
        
        # define variables
        self.w = torch.tensor([-2.]*self.num_tasks_per_batch, requires_grad=True)
        self.alpha = torch.tensor([2.]*self.num_tasks_per_batch, requires_grad=True)
        
        # define optimizer
        self.optimizer_alpha = torch.optim.SGD([self.alpha], lr=.3)
        
        # initialize the statistics of gradients
        self.grad_stats = GradStats(self.w, beta2_rmsprop=self.beta2_rmsprop)
        
        self.o_loss = torch.zeros_like(self.alpha)
        self.o_grad = torch.zeros_like(self.alpha)
        
        self.grad_stats.detach()
        self.ws = [self.w.detach_().requires_grad_()]
        
        #whether in inner optimization
        self.inner_count = 0
        self.outer_count = 0
        
        self.points = []
        self.points_x = []
        self.points_y = []
        
        self.i_loss = self.task.inner_loss(self.ws[-1], self.alpha) + self.weight_decay_w/2. * (self.ws[-1])**2
        self.i_grad = torch.autograd.grad(outputs=self.i_loss, inputs=self.ws[-1], grad_outputs=torch.ones_like(self.i_loss),
                                          create_graph=True, retain_graph=True, only_inputs=True)[0]
        self.grad_stats.update(self.i_grad)
        print('reset')

    def render(self, mode ='human'):
        plt.subplot(2,1,1)
        plt.plot(self.points)

        plt.subplot(2,1,2)
        plt.plot(self.points_x, self.points_y)
        tmp = np.max(self.points_x)
        plt.plot([0, tmp+1], [0, (tmp+1)*self.task.p[0]/2.], 'r--')
        plt.plot([self.task.optimal_alpha[0]], [self.task.optimal_w[0]], 'g^')
        plt.show()
        #plt.savefig("reward.pdf")
    
    def close(self):
        print('close')
