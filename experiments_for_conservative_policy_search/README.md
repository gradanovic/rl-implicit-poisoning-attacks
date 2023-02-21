# Conservative Policy Search Algorithm for Implicit Poisoning Attacks in Two-Agent Reinforcement Learning

This folder contains the source code for "Experiments for Conseervative Policy Section" section of the paper "Implicit Poisoning Attacks in Two-Agent Reinforcement Learning". 

First, go to folder subfolder code.
```
cs code
```

Install all libraries in requirements.txt on your machine to be able to execute the code. You can use the following command line.
```
pip install -r requirements.txt
```
To evaluate different algorithms, you need to execute `main.py` file. You can customize the setting and variables by giving arguments. You can find the explanations for each of the arguments below. 

- `--env`: Select your desired environment from {inventory, navigation}.
- `--var`: Select the variable from {epsilon, influence}.
- `--alg`: Specify the algorithms you want to evaluate. The value of this argument is a string with 4 bits, each of them corresponds to one of the algorithms (the order of algorithms is (Naive, CPS, COPS, UPS)). For example, if you want to evaluate CPS and COPS, but not the other ones, you can use `--alg=0110`.
- `--n_iter`: Select the number of iterations for main loop of the algorithms.
- `--util_iter`: Select the number of iterations for approximately solving bellman equations.
- `--eps`: Select the default value of epsilon when the variable is influence.
- `--time_eval`: Set this parameter to True, if you want to evaluate the time of executing of an algorithm.
- `--time_eval_iter`: Set the number of runs for evaluating an algorithm.
- `--address`: Select the final destination where the results will be stored to.

Then, you can execute the code by running this command line:
```
python main.py <arguments>
```
Note that you can change `main.py` itself, to avoide passing arguments in each run. In that case, it suffices to execute the main:
```
python main.py
```
After executing `main.py`, you will be asked to enter values for variables + hyperpatamerets of each algorithm. Here is a sample of command lines.

Enter values for the variable, separated by space.
```
enter values for the desired variable:
0.01 0.03 0.05 0.07 0.09
```

Enter pairs of (lam,delta) (note that there is no space between lambda and delta), separated by space.
```
enter (lam,delta) values for CPS separated by space:
(20,0.01)
```

```
enter delta values for COPS separated by space:
0.01
```

```
enter lam values for UPS separated by space:
20
```
