

if __name__ == "__main__":
    from Adversarial1DGames.Framework import Framework
else: 
    from Framework import Framework

class Chase1D(Framework):
    def __init__(self, num_timesteps : int = 1000, adv_dist_coef = 0):
        self.adv_dist_coef = adv_dist_coef
        self.num_timesteps = num_timesteps
        super().__init__(length = 5, allow_noop = True, num_goals = 1, speed = 0.25, spawnRange = 1)

    def reset(self):
        self.goal_pos = self.spawn_goal_at([3])[0]
        self.spawn_agent_at(0)
        self.spawn_adversary_at(0.5)
        self.steps = 0
        return self.return_data()

    def step(self, action):
        self.moveAgent(action["agent_0"])
        self.moveAdversary(action["adversary_0"])
        self.steps += 1

        #reward
        distanceGoal = abs(self.goal_pos - self.agent_pos)
        distanceAdversary = abs(self.adversary_pos - self.agent_pos) < (0.05 * self.length)
        rew = distanceAdversary #* self.adv_dist_coef - distanceGoal 
        return self.return_data(), rew, self.steps > self.num_timesteps, {}

if __name__ == "__main__":
    from Adversarial1DGames.Framework import APE
    from TargetPolicies import MoveToPos
    from stable_baselines3 import PPO 

    #adversary = PPO.load("APE/policies/adversary-0")
    #victim = PPO.load("APE/policies/victim-0")

    test = Chase1D(50, 10)
    obs = test.reset()
    test.step({"agent_0" : 1, "adversary_0" : 0})
    for i in range(100):
        #aAdv, _ = adversary.predict(obs["adversary_0"])
        #aVic, _ = victim.predict(obs["agent_0"])
        target = MoveToPos(2.5, 0.1, obs)
        from time import sleep
        test.render()
        obs, _, _, _ = test.step({"adversary_0" : target, "agent_0" : 1})
        sleep(0.1)