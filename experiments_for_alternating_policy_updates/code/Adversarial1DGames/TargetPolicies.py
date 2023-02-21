import numpy as np

def MoveToPos(targetPos, delta, obs):
    agentPos = obs[-1][0]
    if agentPos < targetPos - delta:
        return 1
    if agentPos > targetPos + delta:
        return 0
    return 2