from regex import W
from stable_baselines3 import PPO 
import torch
from utils import Push2DWithPenalty, moveToPos2D, targetPolicyPush
import gym
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def equiDistance(obs, distr):
    middleLearnerGoal = obs["agent_0"][-1, :2] + 0.5 * obs["agent_0"][-1, 2:4]
    adversaryPos = obs["agent_0"][-1, :2] + obs["agent_0"][-1, 4:]
    return moveToPos2D(adversaryPos, middleLearnerGoal, 0.1, distr, False)


adversary = {
            "pos" : lambda obs : moveToPos2D(-obs["adversary_0"][-1, :2], (0,0), 0.2, False, True),
            "random" : lambda obs : random.randint(0, 4),
            "equi" : lambda obs : equiDistance(obs, False)
            }["equi"]

class PositionPush2D(gym.Env):
    def __init__(self, adversary) -> None:
        super().__init__()
        self.env = Push2DWithPenalty(20, frame_stack = 16)
        
        self.action_space = self.env.action_space("agent_0")
        self.observation_space = self.env.observation_space("agent_0")
        self.adversary = adversary
    
    def reset(self):
        self.obs = self.env.reset()
        return self.obs["agent_0"]

    def step(self, action):
        advAction = adversary(self.obs)
        self.obs, rew, done, info = self.env.step({"agent_0" : action, "adversary_0" : advAction})
        return self.obs["agent_0"], rew["agent_0"], done["agent_0"], info["agent_0"]

    def render(self, mode):
        return self.env.render(mode)


env = PositionPush2D(adversary)
victim = PPO("MlpPolicy", env, verbose = 1)
target = lambda obs : targetPolicyPush(obs[0:], 2.9, 3.1, False)
reference = lambda obs : moveToPos2D(-obs[-1, :2], (0,0), 0.2, True, True)
victim.learn(3 * 1e5)

target = lambda obs : targetPolicyPush(obs[0:], 2.9, 3.1, True)

env = Push2DWithPenalty(20, frame_stack = 16)

adversary = {
            "pos" : lambda obs : moveToPos2D(-obs["adversary_0"][-1, :2], (0,0), 0.2, True, True),
            "random" : lambda obs : torch.ones(5) / 5,
            "equi" : lambda obs : equiDistance(obs, True)
            }["equi"]

dist, cost, total = 0, 0, 0 
for i in range(100):
    obs = env.reset()
    done = False
    while not done:
        if i == 0:
            import matplotlib.pyplot as plt
            img = env.render(mode = "rgb_array")
            fig, ax = plt.subplots(1)
            ax.set_aspect("equal")

            ax.imshow(img)
            distance = (obs["agent_0"][-1, 2:4]**2).sum()
            ax.text(10, 30, f"distance to goal : {distance}")

            plt.savefig(f"screenshots/{total}.jpg")
        
        total += 1
        advPol = adversary(obs)
        advAction = torch.multinomial(advPol, 1).item()
        vicAction, _ = victim.predict(obs["agent_0"])

        vic_pol = victim.policy.get_distribution(torch.tensor(obs["agent_0"]).unsqueeze(0).to(device))
        vic_pol = vic_pol.distribution.probs[0]

        targetAction = target(obs["agent_0"])
        referencePol = reference(obs["adversary_0"])

        obs, rew, done, info = env.step({"agent_0" : vicAction, "adversary_0" : advAction})
        done = done["agent_0"]
        
        
        with torch.no_grad():
            dist += torch.norm(vic_pol - torch.tensor(targetAction).to(device), p = 1).item()
            cost += torch.norm(advPol - torch.tensor(referencePol), p = 1).item()     

print(f"Cost : {cost/total}, dist : {dist/total}")

