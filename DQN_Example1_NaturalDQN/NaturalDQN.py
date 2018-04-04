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
EPSILON = 0.5
GAMMA = 0.9
TARGET_REPLACE_ITER = 10
MEMORY_CAPACITY = 50
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
        
        

class DQN():
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATE*2+2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss() # 均方差
        
    def choose_action(self, x):
        x = Variable(torch.Tensor([x]))
        if np.random.uniform() < EPSILON:
            action_value = self.eval_net.forward(x)
            action = torch.max(action_value, 0)[1].data.numpy()[0]
        else:
            action = np.random.randint(0, N_ACTIONS)
        return action
    
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1
    
    def learn(self):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATE]))
        b_a = Variable(torch.LongTensor(b_memory[:, N_STATE:N_STATE+1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, N_STATE+1:N_STATE+2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_STATE:]))
        
        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r[:,0] + GAMMA * q_next.max(1)[0]
        loss = self.loss_func(q_eval, q_target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        global EPSILON
        if EPSILON < 1 and self.learn_step_counter % 10 == 0:
            EPSILON = EPSILON * 1.1
            
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

if __name__ == '__main__':
    dqn = DQN()
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter)
        
        while not is_terminated:
            A = dqn.choose_action(S)
            S_, R = get_env_feedback(S, A)
            dqn.store_transition(S, A, R, S_)
            
            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()
                
            if S_ == N_STATES-1:
                is_terminated = True
            
            S = S_
            update_env(S, episode, step_counter+1)
            step_counter += 1
            
        
