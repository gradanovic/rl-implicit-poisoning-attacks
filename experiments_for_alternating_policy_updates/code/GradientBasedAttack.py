from stable_baselines3 import PPO 

import torch
from torch import nn 
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from Adversarial1DGames.Push1D import Push1D
from Adversarial1DGames.TargetPolicies import MoveToPos
from utils import moveToPos2D, Push2DWithPenalty, targetPolicyPush
from inventory.InventoryManagement import InventoryManagement
from inventory.policies import target_policy_inventory, reference_policy_inventory
from navigation.navigation import Navigation
from navigation.policies import target_policy_navigation, reference_policy_navigation

import gym

import argparse
import os

from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class VictimEnv(gym.Env):
    def __init__(self,
                env,
                adversary,):
        super().__init__()
        self.adversary = adversary
        self.env = env
        self.action_space = self.env.action_space("agent_0")
        self.observation_space = self.env.observation_space("agent_0")
        self.observation_space

    def reset(self):
        self.advAction = []
        self.obs = self.env.reset()
        return self.obs["agent_0"]

    def step(self, action):
        advAction = self.adversary.predict(self.obs["adversary_0"])
        self.advAction.append(advAction)
        self.obs, rew, done, info = self.env.step({"adversary_0" : advAction, "agent_0" : action})
        return self.obs["agent_0"], rew["agent_0"], done["agent_0"], info["agent_0"]
        
    def render(self):
        self.env.render()

    def close(self):
        self.env.close()


class AdversarialAttack:
    class ActorCriticAdversary(nn.Module):
        def __init__(self, obs_len = 6, num_actions = 3) -> None:
            super().__init__()
            self.obs_len = obs_len 
            self.input = nn.Linear(int(obs_len), 64)
            self.ff1 = nn.Linear(64, 128)
            self.ff2 = nn.Linear(128, 128)
            self.policy1 = nn.Linear(128, 128)
            self.policyOut = nn.Linear(128, num_actions)
            self.critic1 = nn.Linear(128, 64)
            self.criticOut = nn.Linear(64, 1)
        
        def forward(self, x):
            # normalize input
            while x.ndim < 3:
                x = torch.unsqueeze(torch.tensor(x), 0)
            if self.obs_len > 1:
                x.flatten(start_dim = 1)
            x = torch.tensor(x).float().to(device)
            x = F.tanh(self.input(x))
            x = F.tanh(self.ff1(x))
            x = F.tanh(self.ff2(x))

            xPol = F.tanh(self.policy1(x))
            xCritic = F.relu(self.critic1(x))
            return self.policyOut(xPol), self.criticOut(xCritic)

        def predict(self, obs):
            with torch.no_grad():
                policy = F.softmax(self.forward(obs)[0].squeeze(), dim = 0)
            return torch.multinomial(policy, 1).item()


    def __init__(self, target_policy, victim, env, reference_policy, bc_coef = 0.1,
                cost_fn = "CE", gamma = 0.99, omega = -0.025, ent_coef = 0.,  n_steps = 5, eps = 0.) -> None:
        assert cost_fn == "CE" or cost_fn == "KL"
        self.target_policy = target_policy
        self.victim = victim
        self.gamma = gamma
        self.env = env
        self.reference_policy = reference_policy
        self.bc_coef = bc_coef
        self.KL = cost_fn == "KL"
        self.cost_fn = (lambda pol, target : F.kl_div(pol, target)) if cost_fn == "KL" else (lambda pol, target : F.cross_entropy(pol.squeeze(), target))
        self.adversary = self.ActorCriticAdversary(obs_len = np.array(env.observation_space("adversary_0").shape).prod(), num_actions = env.action_space("agent_0").n).to(device)
        self.optimizer = torch.optim.Adam(self.adversary.parameters(), lr = 1e-5)
        self.omega = omega
        self.ent_coef = ent_coef
        self.n_steps = n_steps
        self.eps = eps


    def collect_rollout(self, env, adversary, victim, eps, episodes = 1, use_distance = False, victimClass = None):
        states, rewards, actions_adv, log_prob, log_prob_full, actions_vic, oldProbs, values_adv, reference_policies, followsTarget, dists = [], [], [], [], [], [], [], [], [], [], []
        for _ in range(episodes):
            obs = env.reset()
            done = False
            step = 0
            # collect trajectory
            while not done:
                step += 1
                # add state
                states.append(deepcopy(obs))
                
                #predict actions, policy and values
                a_pol, v_adv = adversary(obs["adversary_0"])
                a_pol = a_pol[0]

                a_pol = F.log_softmax(a_pol)
                values_adv.append(v_adv)
                oldProbs.append(a_pol)
                a_adv = torch.multinomial(torch.exp(a_pol), 1).item()
                actions_adv.append(a_adv)
                log_prob.append(a_pol[0, a_adv])
                log_prob_full.append(a_pol)
                a_vic = victim(obs["agent_0"])
                
                # victim is close to the target policy the target policy(NOTE: not used right now )
                
                global args
                if args.environment == "Inventory" and args.use_dist:
                    a_vic = max(a_adv - obs["adversary_0"], 0)

                if use_distance:
                    target = self.target_policy(obs["agent_0"])
                    vic_pol = victimClass.policy.get_distribution(torch.tensor(obs["agent_0"], device = device).flatten().unsqueeze(0))
                    vic_pol = vic_pol.distribution.probs
                    distance = (1 - vic_pol[0, target]).item()

                actions_vic.append(a_vic)

                # get reference Action
                ref_pol = self.reference_policy(obs["adversary_0"])
                reference_policies.append(ref_pol)

                # take step in the environment
                obs, rew, done, _ = env.step({"adversary_0" : a_adv, "agent_0" : a_vic})
                if use_distance:
                    rew["agent_0"] *= distance
                    dists.append(distance)
                rewards.append(self.gamma ** step * rew["agent_0"] - eps)
                done = done["agent_0"]

        oldProbs = torch.stack(oldProbs)
        trajectory = {"states_adv" : torch.stack([torch.tensor(s["adversary_0"]) for s in states]).to(device) , "states_vic" : torch.stack([torch.tensor(s["agent_0"]) for s in states]).to(device), 
                    "rewards" : rewards, "actions_adv" : actions_adv, "log_probs" : torch.stack(log_prob).to(device = device).squeeze(), "actions_vic" : actions_vic,
                    "oldProbs" : oldProbs, "valuesAdversary" : torch.stack(values_adv),
                    "referencePolicy" : reference_policies, "log_probs_full" : torch.stack(log_prob_full), "followsTarget" : followsTarget, "dists" : dists}

        if self.KL:
            trajectory["referencePolicy"] = torch.stack(trajectory["referencePolicy"]).to(device)
            
        # convert to tensors
        for key in trajectory.keys():
            if not torch.is_tensor(trajectory[key]):
                trajectory[key] = torch.tensor(trajectory[key], device = device)
        trajectory["rewards"] = trajectory["rewards"].float()
        return trajectory

    def compute_advantage(self, trajectory):
        rewards = trajectory["rewards"]
        rewards *= (self.gamma ** torch.arange(0, len(rewards)).to(device))
        returns, returnsAdv = [], []
        for i in torch.arange(0, len(rewards)):
            returns.append(torch.sum(rewards[i-1:]))
            returnsAdv.append(torch.sum(rewards[i-1:]))
        trajectory["returns"] = torch.stack(returns)[:-1]
        trajectory["returnsAdv"] = torch.stack(returnsAdv)[:-1]
        advantagesTarget, advantagesVictim = [], []   
        for t, v in enumerate(trajectory["valuesAdversary"][:-1]):
            # last step used for n-step return
            max_step = min(t + self.n_steps, len(rewards) - 1)
            A_advantage = sum(rewards[t : max_step]) + trajectory["valuesAdversary"][max_step] - trajectory["valuesAdversary"][t]
            advantagesTarget.append(A_advantage)
        
        #advantages for adversary
        trajectory["advantagesAdv"] = torch.stack(advantagesTarget).squeeze()
        #normalize
        trajectory["advantagesAdv"] = (trajectory["advantagesAdv"] - trajectory["advantagesAdv"].mean()) / (trajectory["advantagesAdv"].std() + 1e-8)
        
        # normalize returns
        trajectory["returns"] = (trajectory["returns"] - trajectory["returns"].mean()) / (trajectory["returns"].std() + 1e-8)
        trajectory["returnsAdv"] = (trajectory["returnsAdv"] - trajectory["returnsAdv"].mean()) / (trajectory["returnsAdv"].std() + 1e-8)

    def get_random_states(self, size = 50):
        advStates = np.random.choice(np.arange(0, 5, 0.05), size = size, replace = True)
        advVel = np.random.choice(np.arange(-0.25, 0.25, 0.5), size = size, replace = True)
        vicStates = np.random.choice(np.arange(0, 5, 0.05), size = size, replace = True)
        vicStates = np.minimum(vicStates, advStates)
        vicVel = np.random.choice(np.arange(-0.25, 0.25, 0.05), size = size, replace = True)
        states = np.stack([vicStates, vicVel, np.ones(size) * 2, np.ones(size) * 3, advStates, advVel], -1)
        return states


    def learn(self, epochs = 50, victimTrainSteps = 5000):
        writer = SummaryWriter(log_dir = f"runs/{args.environment}-{args.bc_coef}")
        self.adversary.train()
        for i  in range(epochs):
            for currentVictim in self.victim:
                #lineary increase the bc_coef
                if args.increase_alpha:
                    alpha = (args.alpha * i)/(epochs)
                else:
                    alpha = args.alpha
                currentVictim.learn(victimTrainSteps)
                curLoss, runningRewTarget, runningRewNeighbor = 0, 0, 0
                for j in range(100):
                    ratioTarget, ratioNeighbor = torch.tensor(1), torch.tensor(1)
                    
                    trajectory = self.collect_rollout(self.env, self.adversary, self.target_policy, eps = self.eps)
                    self.compute_advantage(trajectory)
        
                    trajectoryNeighbour = self.collect_rollout(self.env, self.adversary, lambda obs : currentVictim.predict(obs)[0], eps = 0., use_distance = True, victimClass = currentVictim)
                    self.compute_advantage(trajectoryNeighbour)

                    # for logging
                    runningRewTarget  += sum(trajectory["rewards"]) 
                    runningRewNeighbor += sum(trajectoryNeighbour["rewards"])
                    entropy, bc_loss = torch.tensor(0.), torch.tensor(0.)
                    for update_step in range(100):
                            if i + j + update_step > 0:
                                # get ratio between old and new probabilities
                                newProbsTarget, _  = self.adversary(trajectory["states_adv"].clone().detach())
                                newProbsTarget = F.log_softmax(newProbsTarget, dim = 2)
                                # here log probs are used
                                ratioTarget = torch.exp(newProbsTarget - trajectory["oldProbs"].clone().detach()).squeeze()
                                ratioTarget = ratioTarget[torch.arange(0, len(newProbsTarget)), trajectory["actions_adv"].detach()][1:]
                                newProbsNeighbour, _ = self.adversary(trajectoryNeighbour["states_adv"].clone().detach())
                                newProbsNeighbour = F.log_softmax(newProbsNeighbour, dim = 2)
                                ratioNeighbor = torch.exp(newProbsNeighbour - trajectoryNeighbour["oldProbs"].clone().detach()).squeeze()
                                ratioNeighbor = ratioNeighbor[torch.arange(0, len(newProbsTarget)), trajectory["actions_adv"].detach()][1:]
                                
                                #entropy
                                oldProbs = torch.cat([newProbsTarget, newProbsNeighbour])
                                entropy = (oldProbs * torch.exp(oldProbs)).sum(dim = 1).mean()

                            # behavioural cloning loss
                            states = trajectoryNeighbour["states_adv"]
                            policy, _ = self.adversary(states)
                            target = []
                            for state in states:
                                pi_0 = self.reference_policy(state)
                                target.append(pi_0)
                            if self.KL:
                                target = torch.stack(target).to(device).float()
                                pol = torch.log(F.softmax(policy, dim = 2)).squeeze()
                            else:
                                target = torch.tensor(target, device = device)
                                pol = policy
                            bc_loss = self.cost_fn(pol, target)
                        
                            
                            # policy loss
                            lossDagger_1 = ratioTarget * trajectory["advantagesAdv"].clone().detach()
                            lossDagger_2 = torch.clamp(ratioTarget, 0.8, 1.2) * trajectory["advantagesAdv"].clone().detach()
                            lossDagger = torch.min(lossDagger_1, lossDagger_2)

                            lossNeighbour_1 = ratioNeighbor * trajectoryNeighbour["advantagesAdv"].clone().detach()
                            lossNeighbour_2 = torch.clamp(ratioNeighbor, 0.8, 1.2) * trajectoryNeighbour["advantagesAdv"].clone().detach()
                            lossNeighbour = torch.min(lossNeighbour_1, lossNeighbour_2)
                            # minimize if learner already follows target

                            # value loss
                            _, pred = self.adversary(torch.cat([trajectoryNeighbour["states_adv"][:-1], trajectory["states_adv"][:-1]]))
                            target = torch.cat([trajectoryNeighbour["returnsAdv"], trajectory["returnsAdv"]]).clone().detach()
                            valLoss = F.mse_loss(target, pred.squeeze())
                            
                            #policy loss
                            policyLoss = (lossNeighbour - lossDagger).mean() 


                            # combine losses
                            loss = self.bc_coef * policyLoss + alpha * bc_loss + valLoss + self.ent_coef * entropy 
                            curLoss += loss.item()

                            #update policy
                            self.optimizer.zero_grad()
                            loss.backward() 
                            self.optimizer.step()
            print(i, curLoss/1000)
            print("rewards: Target : {} Optimal : {}".format( runningRewTarget/100, runningRewNeighbor/100))
            # reset learner if it is not optimal:
            if runningRewTarget > runningRewNeighbor and args.reset:
                print("reset learner")
                env = self.victim[0].get_env()
                self.victim = [PPO("MlpPolicy", env, verbose = 0, ent_coef=0) for _ in range(args.population)]
                for currentVictim in self.victim:
                    currentVictim.learn(args.prelearn_steps)



            writer.add_scalar("Loss", curLoss/1000, i)
            writer.add_scalar("Reward Target", runningRewTarget/100, i)
            writer.add_scalar("Rewards Optimal", runningRewNeighbor/100, i)
        writer.close()
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Gradient Based Attack against RL-Agents")

    parser.add_argument("--environment", choices = ["Push1D", "Push2D", "Inventory", "Navigation"], default = "Push1D", help = "Environments used for experiments")
    parser.add_argument("--prelearn-steps",  default = 1e5, type = int, help = "number of timesteps the learner will be trained before the adversarial attack")
    parser.add_argument("--bc-coef", default = 0.1, type = float, help = "coeficient for the Behavioural Cloning loss of the Adversary")
    parser.add_argument("--ent-coef", default = 0, type = float, help = "Coefficent for the entropy loss of the Adversary")
    parser.add_argument("--save-img", action = "store_true", help = "Save Images of each timesteps in the the eval-episode (for creating gifs)")
    parser.add_argument("--save-json", action = "store_true", help = "Save data of the run as a json file")
    parser.add_argument("--save-dir", default = "rollouts", type = str, help = "Directory where the json file will be saved")
    parser.add_argument("--epochs", default = 25, type = int, help = "Epochs for adversarial training")
    parser.add_argument("--victim-steps", default = 500, type = int, help = "Amount of timesteps the victim will be trained every epoch")
    parser.add_argument("--adv-dist-coef", default = 0, type = float, help = "Penality coeficent for distance to the adversary")
    parser.add_argument("--increase-alpha", action = "store_true", help = "if activated the bc-coeficient will be increase lineary every epoch to increase exploration")
    parser.add_argument("--eps", type = float, default = 0., help = "epsilon parameter for robustness")
    parser.add_argument("--reset", action = "store_true", help = "reset learner if rew_learner < rew_target")
    parser.add_argument("--population", default = 1, type = int, help = "population of optimal learners")
    parser.add_argument("--frame-stack", default = 1, type = int, help = "stack last n frames, NOTE: only implemented for Push2D")
    parser.add_argument("--alpha", type = float, default = 1, help = "coeficient for the behavioural cloning")
    parser.add_argument("--use-dist", action="store_true")

    args = parser.parse_args()
    print(args)
    
    #create environment
    env = { "Push1D" : Push1D(25, adv_dist_coef = args.adv_dist_coef, frame_stack = args.frame_stack),
            "Push2D" : Push2DWithPenalty(adv_dist_coef = args.adv_dist_coef, frame_stack = args.frame_stack),
            "Navigation" : Navigation(),
            "Inventory" : InventoryManagement()}[args.environment]
    
    referencePolicy = { "Push1D" : lambda obs : 0 if obs[-1][4] > 3 else 1,
                        "Push2D" : lambda obs : moveToPos2D(-obs[-1, :2], (0,0), 0.2, True, True),
                        "Navigation" : reference_policy_navigation,
                        "Inventory" : reference_policy_inventory}[args.environment]

    targetPolicy = {"Push1D" : lambda obs : MoveToPos(2.5, 0.25, obs),
                    "Push2D" : lambda obs, return_dist = False : targetPolicyPush(obs, 2.9, 3.1, return_dist),
                    "Navigation" : target_policy_navigation,
                    "Inventory" : target_policy_inventory}[args.environment]
    
    tempEnv = VictimEnv(env, None)
    victim = [PPO("MlpPolicy", tempEnv, verbose = 0, ent_coef=0) for _ in range(args.population)]
    test = AdversarialAttack(target_policy = targetPolicy, victim = victim, cost_fn = "CE" if args.environment == "Push1D" or args.environment == "Navigation" else "KL",
                            env = env, reference_policy = referencePolicy, bc_coef = args.bc_coef, ent_coef = args.ent_coef, eps = args.eps)

    import time
    start_time = time.time()
    for currentVictim in victim:
        vicEnv = VictimEnv(env, test.adversary)

        currentVictim.set_env(vicEnv)
    

        currentVictim.learn(args.prelearn_steps)

    test.learn(epochs = args.epochs, victimTrainSteps = args.victim_steps)
    print(time.time() - start_time)

    successPol, successPos, total, dist, cost, distance_goal = 0, 0,  0, 0, 0, 0
    rollouts = {}

    targetPolicy = {"Push1D" : lambda obs : MoveToPos(2.5, 0.25, obs),
                    "Push2D" : lambda obs : targetPolicyPush(obs, 2.9, 3.1, True),
                    "Navigation": lambda obs : 1,
                    "Inventory" : target_policy_inventory}[args.environment]

    victim = PPO("MlpPolicy", victim[0].get_env(), verbose = 0, ent_coef=0)
    victim.learn(2 * 1e5)

    for i in range(250):
        obs = env.reset()
        done = False
        while not done:
            #save screenshot for video
            if args.save_img and i == 0:
                if args.environment == "Push2D":
                    import matplotlib.pyplot as plt
                    img = env.render(mode = "rgb_array")
                    fig, ax = plt.subplots(1)
                    ax.set_aspect("equal")

                    ax.imshow(img)
                    distance = (obs["agent_0"][-1, 2:4]**2).sum()
                    ax.text(10, 30, f"distance to goal : {distance}")

                    plt.savefig(f"screenshots/{total}.jpg")
                    
                else:
                    import pygame as pg
                    env.render()
                    pg.image.save(env.screen, f"screenshots/{total}.png")
            total += 1
            # get action of agent and adversary
            a_adv = test.adversary.predict(obs["adversary_0"])
            a_vic, _ = victim.predict(obs["agent_0"])

            # action of target-policy
            target = targetPolicy(obs["agent_0"])

            # policy(probability distribution) of victim
            vic_pol = victim.policy.get_distribution(torch.tensor(obs["agent_0"]).unsqueeze(0).to(device))
            vic_pol = vic_pol.distribution.probs[0]
            adv_pol, _ = test.adversary(obs["adversary_0"])
            adv_pol = F.softmax(adv_pol, dim = 2)
            refAction = referencePolicy(obs["adversary_0"])

            if args.environment == "Push1D":
                cost += (1 - adv_pol.squeeze()[refAction]).item()
                dist += (1 - vic_pol[target]).item()
                #successPos += obs["agent_0"][0] > 2.24 and obs["agent_0"][0] < 2.74 
                successPol += a_vic == target
            
            elif args.environment == "Push2D":
                with torch.no_grad():
                    cost += torch.norm(adv_pol - refAction.to(device), p = 1).item()
                    dist += torch.norm(vic_pol - target.to(device), p = 1).item()
                    distance_goal += (obs["agent_0"][-1, 2:4]**2).sum()
                    # right position if the target action is standing still (i.e. action 0 has a prob. of one)
                    successPol += a_vic == target.argmax()
            
            elif args.environment == "Inventory":
                with torch.no_grad():
                    cost += torch.norm(adv_pol.squeeze() - refAction.to(device), p = 1).item()
                    dist += (1 - vic_pol.squeeze()[target]).item()
                    successPol += a_vic == target
                    distance_goal += obs["agent_0"] + a_vic

            elif args.environment == "Navigation":
                cost += (1 - adv_pol.squeeze()[refAction]).item()
                dist += (1 - vic_pol[target]).item()
                #successPos += obs["agent_0"][0] > 2.24 and obs["agent_0"][0] < 2.74 
                successPol += a_vic == target

            obs, rew, done, info = env.step({"adversary_0" : a_adv, "agent_0" : a_vic})
            done = done["agent_0"]

        rollouts["matchingRate"] = float(successPol/total)
        rollouts["Distance"] = float(dist/total)
        rollouts["Cost"] = float(cost/total)
        rollouts["SpendOnTarget"] = float(successPos/total)
        rollouts["distance to goal"] = float(distance_goal/total)

    if args.save_json:
        import json
        if not args.save_dir in os.listdir():
            os.mkdir(args.save_dir)
        with open(args.save_dir + "/" +  str(args.bc_coef) +  str(args.adv_dist_coef) + str(args.population) + ".json", "x") as f:
            json.dump(rollouts, f)
    
    print(f"matching rate : {successPol/total}\n timesteps spent on target Position : {successPos/total}\n distance {dist/total}\n cost: {cost/total}\n distance to goal: {distance_goal/total}")