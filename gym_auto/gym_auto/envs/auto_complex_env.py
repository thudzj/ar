import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym_auto.envs.optimizers import OPTS, LRS, GradStats
from gym_auto.envs.tasks import Task, random_task
import matplotlib as plt

class AutoComplexEnv(gym.Env):
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
        self.task = random_task(num_tasks_per_batch)
        
        # define variables
        self.w = torch.tensor([-2.]*num_tasks_per_batch, requires_grad=True)
        self.alpha = torch.tensor([2.]*num_tasks_per_batch, requires_grad=True)
        
        # define optimizer
        self.optimizer_alpha = torch.optim.SGD([alpha], lr=.3)
        
        # initialize the statistics of gradients
        self.grad_stats = GradStats(w, beta2_rmsprop=beta2_rmsprop)
        
        self.o_loss = torch.zeros_like(alpha)
        self.o_grad = torch.zeros_like(alpha)
        
        self.grad_stats.detach()
        self.ws = [w.detach_().requires_grad_()]
        
        #whether in inner optimization
        self.inner_count = 0
        self.outer_count = 0
    
    def step(self, action):
        #action: opt, lr
        self.ws.append(OPTS[opt](LRS[lr], self.ws[-1], i_grad, grad_stats))
        self.inner_count = (self.inner_count + 1) % inner_T
        
        reward = 0
        episode_over = False
        
        #whether in inner optimization
        if self.inner_count == 0:
            self.outer_count = self.outer_count + 1
            w = self.ws[-1]
            
            self.o_loss = self.task.outer_loss(w, self.alpha).sum()
            self.optimizer_alpha.zero_grad()
            self.o_loss.backward()
            self.o_grad = self.alpha.grad.clone().detach()
            self.optimizer_alpha.step()
            
            # get reward and update the policy
            reward = task.get_reward(self.alpha)
            # optimizer_policy.zero_grad()
            # policy_loss = -logp * reward.mean()
            # policy_loss.backward()
            # optimizer_policy.step()
            
            print("Training ite", self.outer_count, reward)
        
        else:
            reward = 0
        
        i_loss = task.inner_loss(self.ws[-1], alpha) + self.weight_decay_w/2. * (self.ws[-1])**2
        i_grad = torch.autograd.grad(outputs=i_loss, inputs=self.ws[-1], grad_outputs=torch.ones_like(i_loss),
            create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad_stats.update(i_grad)
        ob = self.ws[-1], i_grad, grad_stats

        if self.outer_count == self.outer_T:
            episode_over = True
        else:
            episode_over = False
        return  ob, reward, episode_over,{}

    def reset(self):
        print('reset')
        
    def render(self, mode ='human'):
        print("Optimal: ", task_new.optimal_alpha, task_new.optimal_w)
        print("Found solution: ", alpha.item(), w.item())
        plt.figure()
        plt.plot(points_x, points_y)
        plt.plot([0, np.max(points_x)+1], [0, (np.max(points_x)+1)*task_new.p/2.], 'r--')
        plt.plot([task_new.optimal_alpha], [task_new.optimal_w], 'g^')
        plt.show()
        plt.savefig("example.pdf")
        
    def close(self):
        print('close')

