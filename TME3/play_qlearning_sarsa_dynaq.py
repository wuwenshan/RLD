# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 09:45:01 2020

@author: wuwen
"""

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Qt5agg")

import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
import random
from tqdm import tqdm

from qlearning import Q_Learning
from sarsa import Sarsa
from dynaq import Dyna_Q


def play(agentType="qlearning", worldNumber=0, eps=0.1, alpha=0.001, gamma=0.999):
    # env.action_space : ens des actions possibles
    # env.action_space.n : nombre d'actions possibles
    # env.observation_space : ens des états possibles
    # env.observation_space : nombre d'états possibles

    env = gym.make("gridworld-v0") # Init un environnement

    # setPlan(arg1, arg2)
    # arg1 : fichier de la carte à charger
    # arg2 : liste de récompenses associées aux différents types de cases du jeu 
    env.setPlan("gridworldPlans/plan"+str(worldNumber)+".txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})

    env.verbose = True

    if agentType == "qlearning":
        agent = Q_Learning(env, eps, alpha, gamma)
        
    elif agentType == "sarsa":
        agent = Sarsa(env, eps, alpha, gamma)
        
    elif agentType == "dynaq":
        agent = Dyna_Q(env, eps, alpha, gamma)
        
    else:
        agent = Q_Learning(env, eps, alpha, gamma)
        print("Agent inconnu : qlearning par défaut")

    # Faire un fichier de log sur plusieurs scenarios
    outdir = 'gridworld-v0/random-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    
    #countActions = []
    countRewards = []

    episode_count = 2000
    reward = 0
    done = False
    rsum = 0
    FPS = 0.001

    for i in tqdm(range(episode_count)):
        obs = envm.reset()
        env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
             env.render(FPS)
             env.render(mode="human")
        j = 0
        rsum = 0
        while True:
            action = agent.action(obs, reward)
            obs, reward, done, _ = envm.step(action)
            rsum += reward
            j += 1
            if env.verbose:
                 env.render(FPS)
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                #countActions.append(j)
                countRewards.append(rsum)
                break
                
    np.save("rewards_gridworld_"+str(worldNumber)+"_"+agentType+"_alpha0_1.npy", countRewards)          
    print("Mean & std : ", np.mean(countRewards), np.std(countRewards))
    print("Reward cum : ", np.sum(countRewards))
    print("done")
    env.close()
    
    return countRewards


crQ = play("qlearning", 1)
crS = play("sarsa", 1)
crD = play("dynaq", 1)

crQ = play("qlearning", 5)
crS = play("sarsa", 5)
crD = play("dynaq", 5)

crQ = play("qlearning", 8)
crS = play("sarsa", 8)
crD = play("dynaq", 8)
