import gym
import numpy as np
from gym import spaces
import random

class Navigation(gym.Env):
    def __init__(self, transitionProb = 0.9, maxSteps = 10) -> None:
        super().__init__()
        self.transitionProb = transitionProb
        self.maxSteps = maxSteps
    
    def observation_space(self, agent):
        return {"agent_0": spaces.Discrete(9), "adversary_0" : spaces.Discrete(9)}[agent]

    def action_space(self, agent):
        return {"agent_0" : spaces.Discrete(2), "adversary_0" : spaces.Discrete(2)}[agent]


    def reset(self):
        self.steps = 0
        self.state = np.zeros((1, 1))
        return {"adversary_0" : self.state, "agent_0": self.state.squeeze()}

    def step(self, action):
        self.steps += 1
        rand = random.random()
        if action["agent_0"] == action["adversary_0"] and rand < self.transitionProb:
            takenAction = action["agent_0"] if type(action["agent_0"]) == int else action["agent_0"].item()
        else:
            takenAction = random.randint(0, 1)

        self.state = {
            0 : [0, 1],
            1 : [0, 2],
            2 : [3, 3],
            3 : [1, 4],
            4 : [5, 7],
            5 : [4, 6],
            6 : [5, 6],
            7 : [4, 8],
            8 : [7, 8]
        }[self.state.item()][takenAction]
        self.state = np.reshape(self.state, (1, 1))

        reward = 5 if action["agent_0"] == action["adversary_0"] else -5
        reward += 50 * (self.state.item() == 2)

        done = self.steps >= self.maxSteps

        return ({"adversary_0" : self.state, "agent_0": self.state.squeeze()},
                {"adversary_0" : reward, "agent_0" : reward}, 
                {"adversary_0" : done, "agent_0" : done},
                {"agent_0" : {}, "adversary_0" : {}})

    def render(self):
        print(self.state)


if __name__ == "__main__":
    env = Navigation()
    env.reset()
    env.render()
    done = False
    while not done:
        a1, a2 = np.random.randint(0, 2, 2)
        print(a1, a2)
        state, rew, done, _ = env.step({"agent_0" : a1, "adversary_0" : a2})
        print(state, rew)
        done = done["agent_0"]
        


