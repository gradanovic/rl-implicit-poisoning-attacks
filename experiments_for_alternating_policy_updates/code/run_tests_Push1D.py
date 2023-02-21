import os
import numpy as np
import argparse
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("dir")

    args = parser.parse_args()

    print(args)

    for alph in np.arange(0, 1.1, 0.1):
        os.system(f"python3 GradientBasedAttack.py --environment Push1D --epochs 50 --bc-coef {alph} --adv-dist-coef 0 --save-json --save-dir {args.dir} --prelearn-steps 500000  --victim-steps 5000 --frame-stack 1 --increase-alpha")
