# Reacher-Udacity-Deep-RL-project
DDPG for Unity ML-Agents Reacher environment

Install
--------------------------------------------------------------------------------
We use:
- [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
  to setup the environment,
- and python 3.7

Setup our environment:
```bash
conda --version

# Clone the repo
git clone https://github.com/KhalilGorsan/Reacher-Udacity-Deep-RL-project.git
cd Reacher-Udacity-Deep-RL-project

# Create a conda env
conda env create -f environment.yml

source activate deeprl_udacity

# Install pre-commit hooks
pre-commit install
```
Don't forget to add The Reacher.app unity environment in the root of the project.

To install an already built environment for you, you can download it from one
of the links below. You need only to select the environment that matches your operating
system and unzip it:

Version 1: One (1) Agent
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

Version 2: Twenty (20) Agents
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

Environment
--------------------------------------------------------------------------------
For this project, we will work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![](https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif)

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

Training
--------------------------------------------------------------------------------
To train you _`ddpg`_ agent on the reacher environment, you can run this code
```bash
python train.py
```

After the train is completed, it will provide you the checkpoints of the **actor** and **critic** when the environment is solved as well as a plot of the reward.

Note that This task is episodic, and in order to solve the environment, your agent must get an average score of **+30** over **100** consecutive episodes.

