# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 14:26:08 2020

@author: wuwen
"""

from utils import NN
import matplotlib
import matplotlib.pyplot as plt

#○matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from collections import deque
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from memory import Memory



# DQN with Experience Replay & Target Network
class DQN():
    
    def __init__(self, inSize, outSize, layers, envm, episode_count, batch_size, capacity, eps=0.001, gamma=0.999):
        # Q : premier réseau avec poids random
        self.Q = NN(inSize, outSize, layers) 
        
        # Q hat : deuxième réseau avec les mêmes poids que Q
        self.Q_hat = NN(inSize, outSize, layers)
        self.Q_hat.load_state_dict(self.Q.state_dict()) # copie les poids de Q
        self.memory = Memory(capacity)

        self.envm = envm
        self.episode_count = episode_count
        self.batch_size = batch_size
        self.capacity = capacity
        self.eps = eps
        self.gamma = gamma
        self.C = 10
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.Q.parameters(), lr=0.001)
        
        
    def fit(self):
        
        c = 1
        countRewards = []
        countLoss = []
        
        for i in tqdm(range(self.episode_count)):
            
            obs = self.envm.reset()
            rsum = 0
            done = False
            lsum = 0
            count = 0
            
            while not done:
                count += 1
                action = self.action(obs)
                new_obs, reward, done, _ = env.step(action)
                
                rsum += reward

                target = self.setTarget(new_obs, reward, done)
                self.memory.store([obs, action, reward, new_obs, target])
                    
                if c >= self.batch_size:
                    self.optimizer.zero_grad()
                    idx, w, minibatch = self.memory.sample(self.batch_size)

                    x = torch.tensor([mb[0] for mb in minibatch], dtype=torch.float32, requires_grad=True)
                    y = torch.tensor([mb[4] for mb in minibatch], dtype=torch.float32)

                    
                    y_pred = self.Q(x)
                    #loss = F.smooth_l1_loss(torch.max(y_pred, 1)[0], y)
                    loss = self.loss(torch.max(y_pred, 1)[0], y)
                    
                    
                    loss.backward()
                    self.optimizer.step()  
                    #countLoss.append(loss)
                    lsum += loss.item()
                
                    
                    if done:
                        writer.add_scalar("CartPole/DQN/Reward", rsum, i)
                        countRewards.append(rsum)
                        countLoss.append(lsum / count)
                        
                    if c % self.C == 0:
                        self.Q_hat.load_state_dict(self.Q.state_dict())
                    
                c += 1
                obs = new_obs
                
                
                    
            self.eps *= 0.999
            
                
        return countRewards, countLoss
                
                
    def action(self, obs):
        if np.random.random() < self.eps:
            action = np.random.randint( self.envm.action_space.n )
        else:
            with torch.no_grad():
                action = torch.argmax( self.Q( torch.tensor(obs, dtype=torch.float32 ) ) ).numpy()
            
        return action
    

    def setTarget(self, state, reward, done):
        if done:
            return reward
        else:
            with torch.no_grad():
                x = torch.max( self.Q_hat( torch.tensor(state, dtype=torch.float32) ).detach() )
                return reward + self.gamma * x
                
        
        
seed = 1

env = gym.make('CartPole-v0')
random.seed(seed)
np.random.seed(seed)
env.seed(seed)
torch.manual_seed(seed)
# Enregistrement de l'Agent

inSize = 4
outSize = 2
layers = 200

outdir = 'cartpole-v0/random-agent-results'
envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)

episode_count = 1000
batch_size = 200
capacity = 300

writer = SummaryWriter()

agent = DQN(inSize, outSize, [layers], envm, episode_count, batch_size, capacity)

cr, cl = agent.fit()

#np.save("cartpole_dqn.npy", cr)

print(np.sum(cr))



