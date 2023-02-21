from stable_baselines3 import PPO 
import gym 
from Adversarial1DGames.Push1D import Push1D
from Adversarial1DGames.TargetPolicies import MoveToPos
import torch
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BaselineEnvPos(gym.Env):
    def __init__(self) -> None:
        super().__init__()
        self.env = Push1D(50)

        self.action_space = self.env.action_space("agent_0")
        self.observation_space = self.env.observation_space("agent_0")

    def reset(self):
        self.obs = self.env.reset()
        return self.obs["agent_0"]

    def step(self, action):
        advAction = random.randint(0, 2)
        self.obs, rew, done, info = self.env.step({"agent_0" : action, "adversary_0" : advAction})
        return self.obs["agent_0"], rew["agent_0"], done["agent_0"], info


env = BaselineEnvPos()
victim = PPO("MlpPolicy", env, verbose = 1)
target = lambda obs : MoveToPos(2.5, 0.1, obs)
victim.learn(5*1e4)

#eval
dist, cost, total = 0, 0, 0 
for i in range(100):
    obs = env.reset()
    done = False
    while not done:
        total += 1
        advAction = random.randint(0, 2)
        vicAction, _ = victim.predict(obs)

        vic_pol = victim.policy.get_distribution(torch.tensor(obs).unsqueeze(0).to(device))
        vic_pol = vic_pol.distribution.probs[0]

        obs, rew, done, info = env.step(vicAction)
        
        targetAction = target(obs)
        dist += 1 - (vic_pol[targetAction])
        cost += advAction != (0 if obs[0][4] > 3 else 1)

print(f"Cost : {cost/total}, dist : {dist/total}")