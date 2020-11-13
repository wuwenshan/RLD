import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Qt5agg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy


class PolicyIteration(object):
    """The world's simplest agent!"""

    def __init__(self, action_space, statedic, mdp, gamma=0.99, eps=0.01):
        self.statedic = statedic
        self.mdp = mdp
        self.gamma = gamma
        self.eps = eps
        self.V = {s : np.random.random() for s in statedic.keys()}
        self.pi = {s : np.random.randint(0, action_space.n) for s in statedic.keys()}

    def fit(self):

        ft_V = True
        ft_pi = True
        c = 0
        while ft_pi:
            c += 1
            pi_copy = copy.deepcopy(self.pi)
            
            while ft_V:
                V_copy = copy.deepcopy(self.V)
                
                for state, dict2 in self.mdp.items():
    
                    V_inter = [ np.sum([p*(rew + self.gamma*V_copy[s_dest]) 
                        for p, s_dest, rew, _ in tuples]) 
                            for action, tuples in dict2.items() ]
                    
                    self.V[state] = np.max(V_inter)
                    
                if np.linalg.norm(np.array(list(self.V.values())) - np.array(list(V_copy.values()))) <= self.eps:
                    ft_V = False
        
            for state, dict2 in self.mdp.items():
                
                V_inter = [ np.sum([p*(rew + self.gamma*self.V[s_dest]) 
                    for p, s_dest, rew, _ in tuples]) 
                        for action, tuples in dict2.items() ]
    
                self.pi[state] = np.argmax(V_inter)
                
            if pi_copy == self.pi:
                # print("IDEM PI : ", c)
                ft_pi = False

    def act(self, observation, reward, done):
            
        return self.pi[gridworld.GridworldEnv.state2str(observation)]

        
    
    


if __name__ == '__main__':


    env = gym.make("gridworld-v0")
    env.setPlan("gridworldPlans/plan2.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})

    env.seed(0)  # Initialise le seed du pseudo-random
    # print(env.action_space)  # Quelles sont les actions possibles
    # print(env.step(1))  # faire action 1 et retourne l'observation, le reward, et un done un booleen (jeu fini ou pas)
    env.render()  # permet de visualiser la grille du jeu 
    env.render(mode="human") #visualisation sur la console
    statedic, mdp = env.getMDP()  # recupere le mdp : statedic
    # print("Nombre d'etats : ",len(statedic))  # nombre d'etats ,statedic : etat-> numero de l'etat
    state, transitions = list(mdp.items())[0]
    # print(state)  # un etat du mdp
    # print(transitions)  # dictionnaire des transitions pour l'etat :  {action-> [proba,etat,reward,done]}

    # Execution avec un Agent
    agent = PolicyIteration(env.action_space, statedic, mdp)
    agent.fit()

    episode_count = 1000
    reward = 0
    done = False
    rsum = 0
    FPS = 0.0001
    total_reward = []
    
    for i in range(episode_count):
        obs = env.reset()
        env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render(FPS)
        j = 0
        rsum = 0
        while True:
            action = agent.act(obs, reward, done)
            obs, reward, done, _ = env.step(action)
            rsum += reward
            j += 1
            if env.verbose:
                env.render(FPS)
            if done:
                total_reward.append(rsum)
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break
    print("Reward moyen & std : ", np.mean(total_reward), np.std(total_reward))
    print("done")
    env.close()