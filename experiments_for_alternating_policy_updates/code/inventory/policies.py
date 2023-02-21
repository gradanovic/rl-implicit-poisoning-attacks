import numpy as np
import torch

def target_policy_inventory(state):
    if state > 7:
        return 0
    return 7 - state

def reference_policy_inventory(state):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #TODO: try different reference policies
    return torch.tensor([0.1] * 10, dtype = torch.float)