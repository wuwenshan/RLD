import matplotlib

#matplotlib.use("TkAgg")
import pandas as pd
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
from valueIteration import ValueIteration
from policyIteration import PolicyIteration


if __name__ == '__main__':
    
    env = gym.make("gridworld-v0")
    env.setPlan("gridworldPlans/plan0.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
    env.verbose = True
    statedic, mdp = env.getMDP()  # recupere le mdp : statedic
    
    print("action space : ", env.action_space.n, env.observation_space.n)
    
    
    #agent = ValueIteration(env.action_space, statedic, mdp)
    
    agent = PolicyIteration(env.action_space, statedic, mdp)

    
    # Faire un fichier de log sur plusieurs scenarios
    outdir = 'gridworld-v0/random-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)

    print("obs : ", env.reset())
    
    """

    episode_count = 10000
    reward = 0
    done = False
    rsum = 0
    FPS = 0.001
    for i in range(episode_count):
        obs = envm.reset()
        env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render(FPS)
            env.render(mode="human")
        j = 0
        rsum = 0
        while True:
            action = agent.act(obs, reward, done)
            obs, reward, done, _ = envm.step(action)
            rsum += reward
            j += 1
            if env.verbose:
                env.render(FPS)
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break

    print("done")
    env.close()
    
    
    """
