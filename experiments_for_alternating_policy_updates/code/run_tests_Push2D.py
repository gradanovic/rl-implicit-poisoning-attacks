import os
import numpy as np
import argparse
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("dir")

    args = parser.parse_args()

    print(args)

    for alph in np.arange(0, 5.1, 0.25):
        os.system(f"python3 GradientBasedAttack.py --environment Push2D --epochs 20 --bc-coef {alph} --adv-dist-coef 20 --save-json --save-dir {args.dir} --prelearn-steps 500000  --victim-steps 5000 --frame-stack 16")
