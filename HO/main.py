import numpy as np
import torch
from tasks import Task, random_task
from optimizers import OPTS, LRS, GradStats
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # hyper-parameters
    weight_decay_w = 0
    beta2_rmsprop = 0.9
    outer_T = 100
    inner_T = 5
    num_tasks_per_batch = 1

    # initialize the policy network and the corresponding optimizer
    # policy_net = MLP(20)
    # optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=.01)

    # initialize a task
    task = Task(p=np.array([1]), q=np.array([3]))

    # define variables
    w = torch.tensor([-2.]*num_tasks_per_batch, requires_grad=True)
    alpha = torch.tensor([2.]*num_tasks_per_batch, requires_grad=True)

    # define optimizer
    optimizer_alpha = torch.optim.SGD([alpha], lr=.3)

    # initialize the statistics of gradients
    grad_stats = GradStats(w, beta2_rmsprop=beta2_rmsprop)

    o_loss = torch.zeros_like(alpha)
    o_grad = torch.zeros_like(alpha)

    for outer_ite in range(outer_T):

        grad_stats.detach()
        ws = [w.detach_().requires_grad_()]
        for inner_ite in range(inner_T):
            i_loss = task.inner_loss(ws[-1], alpha) + weight_decay_w/2. * (ws[-1])**2
            i_grad = torch.autograd.grad(outputs=i_loss, inputs=ws[-1], grad_outputs=torch.ones_like(i_loss),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
            grad_stats.update(i_grad)

            # choose optimizer and lr with a policy net
            # policy_o = policy_net(torch.stack([o_loss, o_grad, i_loss, i_grad], 1))
            # opt, lr = policy_o.sample()
            # logp = logp(opt | policy_o) + logp(lr | policy_o)
            opt, lr = 5, 0
            ws.append(OPTS[opt](LRS[lr], ws[-1], i_grad, grad_stats))

        w = ws[-1]

        o_loss = task.outer_loss(w, alpha).sum()
        optimizer_alpha.zero_grad()
        o_loss.backward()
        o_grad = alpha.grad.clone().detach()
        optimizer_alpha.step()

        # get reward and update the policy
        reward = 0 #task.get_reward(alpha)
        # optimizer_policy.zero_grad()
        # policy_loss = -logp * reward.mean()
        # policy_loss.backward()
        # optimizer_policy.step()

        print("Training ite", outer_ite, reward)

    ## test phase: test the trained policy network
    print("Testing")
    task_new = Task(p=np.array([1]), q=np.array([3]))

    w = torch.tensor([-2.], requires_grad=True)
    alpha = torch.tensor([2.], requires_grad=True)
    optimizer_alpha = torch.optim.SGD([alpha], lr=.3)

    grad_stats = GradStats(w, beta2_rmsprop=beta2_rmsprop)

    o_loss = torch.zeros_like(alpha)
    o_grad = torch.zeros_like(alpha)

    points_x, points_y = [alpha.item()], [w.item()]

    for outer_ite in range(outer_T):

        grad_stats.detach()
        ws = [w.detach_().requires_grad_()]
        for inner_ite in range(inner_T):
            
            i_loss = task_new.inner_loss(ws[-1], alpha) + weight_decay_w/2. * (ws[-1])**2
            i_grad = torch.autograd.grad(outputs=i_loss, inputs=ws[-1], grad_outputs=torch.ones_like(i_loss),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
            grad_stats.update(i_grad)

            # choose optimizer and lr with a policy net
            # policy_o = policy_net(torch.stack([o_loss, o_grad, i_loss, i_grad], 1))
            # opt, lr = policy_o.sample()
            opt, lr = 5, 0
            ws.append(OPTS[opt](LRS[lr], ws[-1], i_grad, grad_stats))
            print(i_loss, i_grad, ws[-1])

        w = ws[-1]

        o_loss = task_new.outer_loss(w, alpha).sum()
        optimizer_alpha.zero_grad()
        o_loss.backward()
        o_grad = alpha.grad.clone().detach()
        optimizer_alpha.step()

        points_x.append(alpha.item())
        points_y.append(w.item())

    print("Optima: ", task_new.optimal_alpha, task_new.optimal_w)
    print("Found solution: ", alpha.item(), w.item())
    print(task_new.get_reward(alpha, w))
    plt.figure()
    plt.plot(points_x, points_y)
    plt.plot([0, np.max(points_x)+1], [0, (np.max(points_x)+1)*task_new.p/2.], 'r--')
    plt.plot([task_new.optimal_alpha], [task_new.optimal_w], 'g^')
    plt.show()
    #plt.savefig("example.pdf")
