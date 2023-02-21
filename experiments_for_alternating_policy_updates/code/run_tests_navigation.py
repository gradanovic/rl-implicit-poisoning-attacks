import os
import numpy as np
import argparse
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("dir")
    parser.add_argument("--start", default = 0, type = float)
    parser.add_argument("--stop", default = 1.1, type = float)
    parser.add_argument("--step", default = 0.1, type = float)

    args = parser.parse_args()

    print(args)

    for alph in np.arange(args.start, args.stop, args.step):
        os.system(f"python3 GradientBasedAttack.py --environment Navigation --epochs 25 --bc-coef {alph} --adv-dist-coef 0 --prelearn-steps 50000 --frame-stack 1 --alpha 0.1 --ent-coef 0 --victim-steps 5000 --save-json --save-dir {args.dir} --increase-alpha")
