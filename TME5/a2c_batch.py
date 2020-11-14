
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


learning_rate = 3e-4

# Constants
GAMMA = 0.99
num_steps = 300
max_episodes = 3000


# Deuxième version : avec batch
class A2C_Batch():
    
    def __init__(self, inSize, outSize, hiddenSize, envm, episode_count, gamma=0.99):
        
        self.actor_critic = ActorCritic(inSize, outSize, hiddenSize)
        
        self.envm = envm
        self.episode_count = episode_count
        self.gamma = gamma
        
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        
    def fit(self):
        all_rewards = []
        
        for episode in tqdm(range(self.episode_count)):
            values = []
            rewards = []
            log_probs = []
            state = env.reset()

            for steps in (range(num_steps)):
                value, policy_dist = self.actor_critic( torch.tensor(state, dtype=torch.float32, requires_grad=True ) )
                dist = torch.distributions.Categorical(probs=policy_dist)
                action = dist.sample()
                new_state, reward, done, _ = env.step(action.numpy())


                log_prob = torch.log(policy_dist[action.numpy()])

                rewards.append(reward)
                values.append(value)
                log_probs.append(log_prob)
                #entropy_term += entropy
                state = new_state

                if done or steps == num_steps-1:
                    Qval, _ = self.actor_critic( torch.tensor(new_state, dtype=torch.float32, requires_grad=True ) )
                    writer.add_scalar("CartPole/A2CBatch/Reward", np.sum(rewards), episode)
                    all_rewards.append(np.sum(rewards))
                    break

            values = torch.FloatTensor(values)

            # compute Q values
            Qvals = torch.zeros_like(values)
            for t in reversed(range(len(rewards))):
                Qval = rewards[t] + GAMMA * Qval
                Qvals[t] = Qval

            Qvals = torch.FloatTensor(Qvals)
            log_probs = torch.stack(log_probs)
            
            advantage = Qvals - values
            ac_loss = (log_probs * advantage).mean()
            
            self.optimizer.zero_grad()
            ac_loss.backward()
            self.optimizer.step()
            
        return all_rewards
    
    
env = gym.make('CartPole-v0')

# Enregistrement de l'Agent
inSize = 4
outSize = 2
layers = [64,32]

outdir = 'cartpole-v0/random-agent-results'
envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
writer = SummaryWriter()

episode_count = 10000

agent = A2C_Batch(inSize, outSize, layers, env, episode_count)

cr = agent.fit()
