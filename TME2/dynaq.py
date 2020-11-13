# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 10:14:37 2020

@author: wuwen
"""

import numpy as np
import random

class Dyna_Q():
    
    def __init__(self, env, eps, alpha, gamma):
        self.Q = {}
        self.model = {}
        self.env = env
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma
        self.last_state = None
        self.last_action = None
    
    def action(self, obs, rew):
        
        state = self.env.state2str(obs)
        
        self.Q.setdefault(state, [0, 0, 0, 0]) # rajoute l'état s'il n'est pas présent sinon ne fait rien
        
        # with epsilon-greedy approach
        if np.random.random() < 1 - self.eps:
            action = np.argmax( self.Q[ state ] )
            
        else:
            action = np.random.randint( self.env.action_space.n )
            
        self.updateQ(action, state, rew)
        
        return action
    
            
    
    def updateQ(self, action, state, rew):
        
        if self.last_state == None and self.last_action == None:
            self.last_state = state
            self.last_action = action
            
        else:
            self.Q[self.last_state][self.last_action] += self.alpha * ( rew + self.gamma * np.max( self.Q[state] ) - self.Q[self.last_state][self.last_action] ) 
            self.model.setdefault((self.last_state, self.last_action), (rew, state))
            for _ in range(100):
                key = random.choice(list(self.model.keys())) # key[0] for the previous state and key[1] for the action taken
                val = self.model[key] # val[0] for the reward and val[1] for the state
                self.Q[key[0]][key[1]] += self.alpha * ( val[0] + self.gamma * np.max( self.Q[val[1]] ) - self.Q[key[0]][key[1]] )

            self.last_state = state
            self.last_action = action
            
        