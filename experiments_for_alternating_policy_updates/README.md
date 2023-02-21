# Gradient Based Implicit Poisoning Attacks in Two-Agent Reinforcement Learning

This folder contains the source code for the section "Experiments for Alternating Policy Updates" of the paper *Implicit Poisoning Attacks in Two-Agent Reinforcement Learning*.

To replicate the results for Push 2D, run GradientBasedAttack.py with the following arguments:
```console 
python3 GradientBasedAttack.py --environment Push2D --epochs 20 --bc-coef [lambda] --adv-dist-coef 20 --save-json --save-dir [directory] --prelearn-steps 500000  --victim-steps 5000 --frame-stack 16
```

for the Push 1D environment:
```console
python3 GradientBasedAttack.py --environment Push1D --epochs 50 --bc-coef [lambda] --adv-dist-coef 0 --save-json --save-dir [directory] --prelearn-steps 500000  --victim-steps 5000 --frame-stack 1 --increase-alpha --alpha 0.1
```
for the inventory management environment:
```console
python3 GradientBasedAttack.py --environment Inventory --epochs 25 --bc-coef [lambda] --adv-dist-coef 0 --prelearn-steps 0 --frame-stack 1 --alpha 0 --ent-coef 0.01 --victim-steps 0 --use-dist
```

for the navigation environment
```console
python3 GradientBasedAttack.py --environment Navigation --epochs 25 --bc-coef [lambda] --adv-dist-coef 0 --prelearn-steps 50000 --frame-stack 1 --alpha 0.1 --ent-coef 0 --victim-steps 5000 --increase-alpha
```


Which will generate a JSON file including the results in the provided directory.

Alternatively, the provided 
```console
run_tests_[Push1D|Push2D|Inventory|navigation].py [directory]
```
script can reproduce the data required for the graphs provided in the Paper. 

The Baselines can be run using the following arguments:

## Random Adversary:
Push1D:
```console
python3 BaselineRandom.py
```
Push2D
```console
python3 Baselines2D.py random
```
Inventory
```console
python3 GradientBasedAttack.py --environment Inventory --epochs 0 --bc-coef 0 --adv-dist-coef 0 --prelearn-steps 0 --frame-stack 1 --alpha 0 --ent-coef 0.01 --victim-steps 0 --use-dist
```
Navigation
```console
python3 NavigationRandom.py
```

## Move to Target Position(1D)
```console
python3 BaselinePosition.py
```

## Equal Distance(2D)
```console
python3 Baseline2D.py equi
```

## Random Learner
Push1D
```console
python3 GradientBasedAttack.py --environment Push1D --epochs 50 --alpha 0.1 --bc-coef [lambda] --adv-dist-coef 0 --save-json --save-dir [directory] --prelearn-steps 0  --victim-steps 0 --frame-stack 25 --increase-alpha"
```
Push2D
```console
python3 GradientBasedAttack.py --environment Push2D --epochs 20 --alpha 1 --bc-coef [lambda] --adv-dist-coef 20 --save-json --save-dir [directory] --prelearn-steps 0  --victim-steps 0 --frame-stack 16
```
Inventory
```console
python3 GradientBasedAttack.py --environment Inventory --epochs 25 --bc-coef [lambda] --adv-dist-coef 0 --prelearn-steps 0 --frame-stack 1 --alpha 0 --ent-coef 0.01 --victim-steps 0
```
Navigation
```console
python3 GradientBasedAttack.py --environment Navigation --epochs 25 --bc-coef [lambda] --adv-dist-coef 0 --prelearn-steps 0 --frame-stack 1 --alpha 0.1 --ent-coef 0 --victim-steps 0 --increase-alpha
```

## Symmetric APU
Push1D
```console
python3 GradientBasedAttack.py --environment Push1D --alpha 0.1 --epochs 50 --bc-coef [lambda] --adv-dist-coef 0 --save-json --save-dir [directory] --prelearn-steps 0  --victim-steps 100 --frame-stack 1 --increase-alpha
```

Push2D
```console
python3 GradientBasedAttack.py --environment Push2D --epochs 100 --alpha 1 --bc-coef [lambda] --adv-dist-coef 20 --save-json --save-dir [directory] --prelearn-steps 0  --victim-steps 100 --frame-stack 16
```

Navigation
```console
python3 GradientBasedAttack.py --environment Navigation --epochs 25 --bc-coef [lambda] --adv-dist-coef 0 --prelearn-steps 0000 --frame-stack 1 --alpha 0.1 --ent-coef 0 --victim-steps 1000
```
Inventory
```console
python3 GradientBasedAttack.py --environment Inventory --epochs 25 --bc-coef [lambda] --adv-dist-coef 0 --prelearn-steps 0 --frame-stack 1 --alpha 0 --ent-coef 0.01 --victim-steps 1000
```

## Distance-only APU
Push1D
```console
python3 GradientBasedAttack.py --environment Push1D --epochs 20 --alpha 0 --bc-coef 1 --adv-dist-coef 0 --save-json --save-dir [directory] --prelearn-steps 500000  --victim-steps 5000 --frame-stack 1 --increase-alpha --alpha 0.1
```

Push2D
```console 
python3 GradientBasedAttack.py --environment Push2D --epochs 20 --bc-coef 1 --adv-dist-coef 20 --save-json --save-dir [directory] --prelearn-steps 500000  --victim-steps 5000 --frame-stack 16 --alpha 0
```

Inventory
```console
python3 GradientBasedAttack.py --environment Inventory --epochs 25 --bc-coef [lambda] --adv-dist-coef 0 --prelearn-steps 0 --frame-stack 1 --alpha 0 --ent-coef 0 --victim-steps 0 --use-dist
```
Navigation
```console
python3 GradientBasedAttack.py --environment Navigation --epochs 25 --bc-coef [lambda] --adv-dist-coef 0 --prelearn-steps 50000 --frame-stack 1 --alpha 0 --ent-coef 0 --victim-steps 5000 --increase-alpha
```
