import pettingzoo
import gym
import numpy as np

from gym import spaces

class InventoryManagement(gym.Env):
    def __init__(self, M = 10, max_steps = 10) -> None:
        super().__init__()
        self.inventory = np.array(0)
        self.M = M
        self.max_steps = max_steps

    def observation_space(self, agent):
        return {"agent_0": spaces.Discrete(self.M), "adversary_0" : spaces.Discrete(self.M)}[agent]

    def action_space(self, agent):
        return {"agent_0" : spaces.Discrete(self.M), "adversary_0" : spaces.Discrete(self.M)}[agent]

    def reset(self):
        self.inventory = np.array(0)
        self.steps = 0
        return {"adversary_0" : np.expand_dims(self.inventory, 0) , "agent_0" : np.expand_dims(self.inventory, 0)}

    def step(self, actions):
        self.steps += 1

        sell = actions["adversary_0"]
        buy = int(actions["agent_0"])

        # get reward
        sellRew = 10 * sell if sell <= buy + self.inventory else 0 
        holdRew = - self.inventory - buy
        buyRew = -(buy > 0) * (4 + 2 * buy)
        reward = (sellRew + holdRew + buyRew)

        if buy + self.inventory >= self.M:
            reward = -100

        # update inventory
        if self.inventory + buy < self.M:
            if sell <= buy + self.inventory:
                self.inventory = buy - sell + self.inventory
            else:
                self.inventory += buy

        return ({"adversary_0" : np.expand_dims(self.inventory, 0), "agent_0" : np.expand_dims(self.inventory, 0)}, 
                {"agent_0" : reward}, 
                {"adversary_0" : self.steps > self.max_steps, "agent_0" : self.steps > self.max_steps},
                {"agent_0" : {}, "adversary_0" : {}})

    def render(self):
        print(self.inventory)

if __name__ == "__main__":
    from random import randint
    from policies import target_policy_inventory
    env = InventoryManagement()

    env.reset()

    for i in range(10):
        actions = {"adversary_0" : randint(0, env.inventory + 3), "agent_0" : randint(0, 9)}
        print(actions)
        _, rew, _, _ = env.step(actions)
        print(rew["agent_0"])
        env.render()
    env.render()