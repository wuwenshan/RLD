# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 09:26:57 2020

@author: wuwen
"""

import matplotlib
import matplotlib.pyplot as plt

#matplotlib.use("TkAgg")
import gym
#import gridworld
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
from torch.autograd import Variable

from nn_ac import ActorCritic
from torch.utils.tensorboard import SummaryWriter


# Première version : résultats médiocres (~10/20 reward)
class A2C():
    
    def __init__(self, inSize, outSize, layers, envm, episode_count, gamma=0.99):
        
        self.model = ActorCritic(inSize, outSize, layers)
        
        self.envm = envm
        self.episode_count = episode_count
        self.gamma = gamma
        self.loss_V = nn.SmoothL1Loss()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
    def fit(self):
        
        countRewards = []
        countLoss = []
        
        for i in tqdm(range(self.episode_count)):
            
            obs = self.envm.reset()
            rsum = 0
            done = False
            
            while not done:
                
                # 1. take action a and get (s, a, s', r)
                value, policy = self.model( torch.tensor(obs, dtype=torch.float32 ) )
                dist = torch.distributions.Categorical(probs=policy)
                action = dist.sample()
                new_obs, reward, done, _ = env.step(action.numpy())
                
                log_prob = torch.log(policy[action.numpy()])
                
                rsum += reward
                
                # 2. update V using target r + gamma * V(s')
                self.optimizer.zero_grad()
                with torch.no_grad():
                    v, _ = self.model( torch.tensor(new_obs, dtype=torch.float32) )
                    V = reward + self.gamma * v
                loss_V = self.loss_V(value, V)
                loss_V.backward()

                
                # 3. evaluate

                A = V - value         
                # compute gradient
                J = log_prob * A.detach()

                # Etape 5
                # Mise à jour des paramètres de PI
                J.backward()
                self.optimizer.step()
                
                obs = new_obs
                
                if done:
                    writer.add_scalar("Cartpole/A2C/Reward", rsum, i)
                    countRewards.append(rsum)
                
        return countRewards
        
    
env = gym.make('CartPole-v0')

# Enregistrement de l'Agent
inSize = 4
outSize = 2
layers = [64,32]

outdir = 'cartpole-v0/random-agent-results'
envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
writer = SummaryWriter()

episode_count = 10000

agent = A2C(inSize, outSize, layers, env, episode_count)

cr = agent.fit()

