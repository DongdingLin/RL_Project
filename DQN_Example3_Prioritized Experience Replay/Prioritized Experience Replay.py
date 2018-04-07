# coding: utf-8
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import sys
import time

np.random.seed(1)
torch.manual_seed(1)

BATCH_SIZE = 16
LR = 0.1
GAMMA = 0.9
TARGET_REPLACE_ITER = 10
MEMORY_CAPACITY = 20
N_ACTIONS = 2
N_STATES = 10
N_STATE = 1
ACTIONS = ['left', 'right']
MAX_EPISODES = 100
FRESH_TIME = 0.3

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATE, 10)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(10, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        action_value = self.out(x)
        return action_value
    
class SumTree(object):
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        
    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, p)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:
            cl_idx = 2 * parent_idx + 1
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]

class Memory(object):
    epsilon = 0.01
    alpha = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))
        pri_seg = self.tree.total_p / n
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon
        abs_errors = abs_errors.data.numpy()
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment
    if S == N_STATES - 1:
        interaction = 'Episode %s: total_steps = %s\n' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def get_env_feedback(S, A):
    # This is how agent will interact with the environment
    if A == 1:    # move right
        if S == N_STATES - 2:   # terminate
            S_ = N_STATES - 1
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:   # move left
        R = 0
        if S == 0:
            S_ = S  # reach the wall
        else:
            S_ = S - 1
    return S_, R

class DQNPrioritizedReplay():
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.learn_step_counter = 0
        self.memory = Memory(capacity=MEMORY_CAPACITY)
        self.memory_counter = 0
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss() # 均方差
        self.epsilon_max = 0.9
        self.e_greedy_increment = None
        self.epsilon_increment = self.e_greedy_increment
        self.epsilon = 0 if self.e_greedy_increment is not None else self.epsilon_max
        
        
    def choose_action(self, x):
        x = Variable(torch.Tensor([x]))
        if np.random.uniform() < self.epsilon:
            action_value = self.eval_net.forward(x)
            action = torch.max(action_value, 0)[1].data.numpy()[0]
        else:
            action = np.random.randint(0, N_ACTIONS)
        return action
    
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        self.memory.store(transition)
        self.memory_counter += 1
        
    
    def learn(self):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        
        tree_idx, batch_memory, ISWeights = self.memory.sample(BATCH_SIZE)
        
        
        b_s = Variable(torch.FloatTensor(batch_memory[:, :N_STATE]))
        b_a = Variable(torch.LongTensor(batch_memory[:, N_STATE:N_STATE+1].astype(int)))
        b_r = Variable(torch.FloatTensor(batch_memory[:, N_STATE+1:N_STATE+2]))
        b_s_ = Variable(torch.FloatTensor(batch_memory[:, -N_STATE:]))

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = torch.unsqueeze((b_r[:,0] + GAMMA * q_next.max(1)[0]), 1)
        
        abs_error = torch.sum(torch.abs(q_target - q_eval), 1)
        self.memory.batch_update(tree_idx, abs_error)
        loss = self.loss_func(q_eval, q_target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

if __name__ == '__main__':
    dqn_PrioritizedReplay = DQNPrioritizedReplay()
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter)
        
        while not is_terminated:
            A = dqn_PrioritizedReplay.choose_action(S)
            S_, R = get_env_feedback(S, A)
            dqn_PrioritizedReplay.store_transition(S, A, R, S_)
            
            if dqn_PrioritizedReplay.memory_counter > MEMORY_CAPACITY:
                dqn_PrioritizedReplay.learn()
                
            if S_ == N_STATES-1:
                is_terminated = True
            
            S = S_
            update_env(S, episode, step_counter+1)
            step_counter += 1
            