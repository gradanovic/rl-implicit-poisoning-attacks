import random
import gym
from gym import spaces
import pygame as pg
import torch
import numpy as np
class Framework(gym.Env):
    def __init__(self,
        length : float = 1,
        allow_noop : bool = True,
        num_goals : int = 0,
        speed : float = 0.3,
        acc : float = 0.1,
        spawnRange : float = 1/5,
        frame_stack : int = 1) -> None:

        self.length = length
        self.allow_noop = allow_noop
        self.num_goals = num_goals

        #speed 
        self.speed = speed
        self.acc = acc

        #spawning
        self.spawnRange = spawnRange

        #actions
        self.allow_noop = allow_noop

        # frame stacking
        self.frame_stack = frame_stack

    def action_space(self, agent):
        return {"agent_0" : spaces.Discrete(3 if self.allow_noop else 2), "adversary_0" : spaces.Discrete(3 if self.allow_noop else 2)}[agent]

    def observation_space(self, agent):
        return {"agent_0" : spaces.Box(-float(self.length), float(self.length), (self.frame_stack, 6)), "adversary_0" : spaces.Box(-float(self.length), float(self.length), (self.frame_stack, 6))}[agent]
    ## utils
    def clip(self, val, low, high):
        return max(min(val, high), low)

    def return_data(self):
        dat = torch.tensor((self.agent_pos, self.agent_speed, *self.goals, self.adversary_pos, self.adversary_speed))
        datAdv = torch.ones(4)
        # relative distance to agent
        datAdv[0] = dat[0]
        # relative distance to goals
        datAdv[1] = dat[3] - dat[2]
        datAdv[2:4] = dat[3:5]

        datVic = torch.ones(4) 
        datVic[0:2] = dat[0:2]
        datVic[2] = dat[0] - dat[2]
        datVic[3] = dat[3]
        return {"agent_0" : dat, "adversary_0" : dat}


    ## spawning
    def spawn_adversary_randomly(self):
        self.adversary_pos = random.choice(np.arange(2.5, self.length, 0.25))
        self.adversary_speed = 0
        return self.adversary_pos

    def spawn_agent_randomly(self):
        self.agent_pos = random.choice(np.arange(0, self.adversary_pos, 0.25))
        self.agent_speed = 0
        return self.agent_pos

    def spawn_goals(self):
        self.goals = [random.random() * self.length for _ in range(self.num_goals)]
        return self.goals

    def spawn_goal_at(self, pos):
        self.goals = [*pos]
        return self.goals

    def spawn_adversary_at(self, pos):
        self.adversary_pos = pos
        self.adversary_speed = 0
        return pos
    
    def spawn_agent_at(self, pos):
        self.agent_pos = pos
        self.agent_speed = 0
        return pos

    #movement 
    def moveAgent(self, action):
        curAcc = [-self.acc, self.acc, -self.agent_speed][action]
        self.agent_speed = self.clip(self.agent_speed + curAcc, -self.speed, self.speed)
        self.agent_pos = self.clip(self.agent_pos + self.agent_speed, 0, self.length)
        self.agent_pos = self.clip(self.agent_pos, 0, self.adversary_pos)
        return self.agent_pos
        
    def moveAdversary(self, action):
        curAcc = [-self.acc, self.acc, -self.adversary_speed][action]
        self.adversary_speed = self.clip(self.adversary_speed + curAcc, -self.speed, self.speed)
        self.adversary_pos = self.clip(self.adversary_pos + self.adversary_speed, 1, self.length)
        self.agent_pos = self.clip(self.agent_pos, 0, self.adversary_pos)
        return self.adversary_pos
        
    #rendering
    def init_display(self):
        self.screen = pg.display.set_mode((self.length * 250, 100))
    
    def render(self):
        try:
            self.screen
        except:
            self.init_display()
        
        self.screen.fill((255, 255, 255))
        pg.draw.rect(self.screen, (175, 0, 0), pg.Rect(2.45 * 250, 40, 50, 10))
        # draw adversary 
        pg.draw.circle(self.screen, color = (255, 0, 0), 
            center = (self.adversary_pos  * 250, 50), radius = 10)

        # draw agent
        pg.draw.circle(self.screen, color =  (0, 255,0), 
                        center = (self.agent_pos * 250, 50), radius = 10)

        #draw goal
        for goal in self.goals:
            pg.draw.rect(self.screen, color = (255,215,0),
                        rect = pg.Rect(goal *250 - 5, 45, 10, 10))

        pg.display.flip()