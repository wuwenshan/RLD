# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 10:30:16 2020

@author: wuwen
"""

import numpy as np
import gridworld
import copy

class ValueIteration():
    
    def __init__(self, action_space, statedic, mdp, eps=1e5, gamma=0.9):
        self.statedic = statedic
        self.mdp = mdp
        self.action_space = action_space
        self.eps = eps
        self.gamma = gamma
        
    def act(self, obs, reward, done):
        nb_s = len(self.statedic) # Nombre d'états
        V = dict(zip(self.statedic, np.random.random(nb_s)))
        states = [s for s in self.statedic]
        pi = {s: None for s in states}
        ft = True # First Time
        i = 0

        print("PI : ", pi)

        while ft or np.linalg.norm(np.array(list(V.values())) - np.array(list(V_copy.values()))) > self.eps:
            V_copy = copy.deepcopy(V)
            for state, dico in self.mdp.items(): # parcours de tous les états
                
                V_inter = [np.sum([p*(rew+self.gamma*V_copy[s_dest]) 
                        for p, s_dest, rew, _ in tuples]) 
                            for action, tuples in dico.items() ]
                
                V[state] = np.max(V_inter)
                
            i += 1
            ft = False
    
        for state, dico in self.mdp.items(): # parcours de tous les états
                
            V_inter = [np.sum([p*(rew+self.gamma*V[s_dest]) 
                        for p, s_dest, rew, _ in self.mdp[state][action]]) 
                            for action, tuples in dico.items() ]
            #print("arg : ", np.argmax(V_inter))
            pi[state] = np.argmax(V_inter)
    
   
            
        return pi[gridworld.GridworldEnv.state2str(obs)]
            
