import random
import numpy as np
import torch
from Adversarial1DGames.Framework import Framework
if __name__ == "__main__":
    from Adversarial1DGames.Framework import Framework

class Push1D(Framework):
    def __init__(self, num_timesteps : int = 1000, adv_dist_coef = 0, teleport_prob = 0.1, frame_stack = 1):
        self.adv_dist_coef = adv_dist_coef
        self.num_timesteps = num_timesteps
        self.teleport_prob = teleport_prob
        super().__init__(length = 5, allow_noop = True, num_goals = 1, speed = 0.25, frame_stack = frame_stack)

    def create_buffer(self):
        dat = self.return_data()
        self.frame_buffer = {"agent_0" : torch.stack([dat["agent_0"] for _ in range(self.frame_stack)]),
                "adversary_0" : torch.stack([dat["adversary_0"] for _ in range(self.frame_stack)])}

    def update_buffer(self):
        dat = self.return_data()
        self.frame_buffer["agent_0"] = torch.cat([self.frame_buffer["agent_0"], dat["agent_0"].unsqueeze(0)])[1:]
        self.frame_buffer["adversary_0"] = torch.cat([self.frame_buffer["adversary_0"], dat["adversary_0"].unsqueeze(0)])[1:]

    def reset(self):
        self.goal_pos = self.spawn_goal_at([0, 3])
        self.spawn_adversary_randomly()
        self.spawn_agent_randomly()
        self.adversary_pos = self.clip(self.adversary_pos, 2.5, 5)
        self.steps = 0
        self.create_buffer()
        return self.frame_buffer

    def step(self, action):
        self.moveAgent(action["agent_0"])
        self.moveAdversary(action["adversary_0"])
        self.steps += 1
        
        #teleport learner randomly
        if random.random() < self.teleport_prob:
            self.spawn_agent_randomly()

        #reward
        distanceGoal2 = (self.goals[1] - self.agent_pos)**2
        #distanceAdversary = abs(self.adversary_pos - self.agent_pos) < 0.05
        rew = - (distanceGoal2)
        # distance to adversary
        #rew -= self.adv_dist_coef * distanceAdversary
        
        # rew -= -0.01 * (action in [0,1]) 

        self.update_buffer()

        #dictonary for combatibility with petting zoo
        return self.frame_buffer, {"agent_0" : rew}, {"agent_0" : self.steps >= self.num_timesteps}, {"agent_0" : {}}