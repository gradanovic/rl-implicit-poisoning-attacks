import numpy as np
import torch

from pettingzoo.mpe import simple_adversary_v2
import gym
from gym.spaces import Box

import random

class Push2DWithPenalty(gym.Env):
    def __init__(self, adv_dist_coef = 1, max_dist_rew = 0.5, move_penalty = 0,  frame_stack = 1) -> None:
        super().__init__()
        self.adv_dist_coef = adv_dist_coef
        self.env = simple_adversary_v2.parallel_env(N = 1, max_cycles = 40)
        self.max_dist_rew = max_dist_rew
        self.move_penalty = move_penalty
        self.frame_stack = frame_stack
        self.frame_buffer = {}

    def create_buffer(self, state):
        self.frame_buffer = {"agent_0" : np.array([state["agent_0"] for _ in range (self.frame_stack)]), "adversary_0" : np.array([state["adversary_0"] for _ in range(self.frame_stack)])}

    def update_buffer(self, state):
        self.frame_buffer["agent_0"] = np.append(self.frame_buffer["agent_0"], [state["agent_0"]], axis = 0)[1:]
        self.frame_buffer["adversary_0"] = np.append(self.frame_buffer["adversary_0"], [state["adversary_0"]], axis = 0)[1:]

    def action_space(self, agent):
        return self.env.action_space(agent)

    def observation_space(self, agent):
        if agent == "adversary_0":
            return Box(-float("inf"), float("inf"), (self.frame_stack, 5))
        else: 
            return Box(-float("inf"), float("inf"), (self.frame_stack, 6))

    def reset(self):
        obs = self.env.reset()
        distGoal = obs["agent_0"][2:4]
        distGoal = ((distGoal ** 2).sum()) **(1/2)
        obs["adversary_0"] = np.append(obs["adversary_0"], distGoal)
        self.create_buffer(obs) 
        return self.frame_buffer

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        distAdv = obs["agent_0"][-2:]

        distAdv = (((distAdv ** 2).sum()) ** (1/2)) < self.max_dist_rew
        distAdv *= self.adv_dist_coef

        distGoal = obs["agent_0"][2:4]
        distGoal = ((distGoal ** 2).sum()) **(1/2)

        #add distance to observation
        obs["adversary_0"] = np.append(obs["adversary_0"], distGoal)

        rew = {"agent_0" : - distGoal - distAdv}
        #add move punishment to encourage no-op
        rew["agent_0"] -= 0 if action["agent_0"] == 0 else self.move_penalty

        self.update_buffer(obs)

        return self.frame_buffer, rew, done, info

    def render(self, mode):
        return self.env.render(mode)

def masked_softmax(vec, mask):
    exp = vec
    # apply mask
    exp *= mask
    sum = exp.sum()
    return exp / sum


def moveToPos2D(agentPos, targetpos, d, return_distribution = False, isDistance = False):
    if not isDistance:
        xPos, yPos = agentPos
        xTarget, yTarget = targetpos
        distance = ((xPos - xTarget)**2 + (yPos - yTarget)**2) ** (1/2)
        xDist = (xPos - xTarget).item()
        yDist = (yPos - yTarget).item()
    else:
        if torch.is_tensor(agentPos):
            agentPos = np.array(agentPos.cpu())
        distance = ((agentPos ** 2).sum()) ** (1/2)
        xDist = agentPos[0]
        yDist = agentPos[1]
    if distance < d:
        return torch.tensor([1, 0, 0, 0 ,0]).float() if return_distribution else 0
    else: 
        res = np.abs(np.array([0, xDist, xDist, yDist, yDist]))
        mask = np.array([0, xDist > 0 , xDist < 0, yDist > 0, yDist < 0])
        res = masked_softmax(res, mask)
    return torch.from_numpy(res) if return_distribution else np.argmax(res).item()

def targetPolicyPush(obs, d_in, d_out, return_distribution = True):
    #only use latest state
    obs = obs[-1]
    assert d_in < d_out
    distance = (obs[2:4] ** 2).sum() ** (1/2)
    if distance < d_in:
        return moveToPos2D(obs[2:4], (0,0), 0, return_distribution, True)
    elif distance > d_out:
        return moveToPos2D(-obs[2:4], (0,0), 0, return_distribution, True)
    else:
        return torch.tensor([1, 0, 0, 0, 0]) if return_distribution else random.randint(0, 4)


if __name__ == "__main__":
    from time import sleep
    test = Push2DWithPenalty(frame_stack=4)
    obs = test.reset()
    for _ in range(25):
        test.render(mode = "rgb_array")
        polVic = targetPolicyPush(obs["agent_0"], 2, 2.5, False)
        polAdv = moveToPos2D(-obs["adversary_0"][-1, :2], (0,0), 0.1, False, True)
        obs, _, _, _ = test.step({"adversary_0" : polAdv, "agent_0" : polVic})
        sleep(0.1)