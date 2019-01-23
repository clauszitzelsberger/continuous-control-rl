# Continuous Control
Udacity's Deep Reinforcement Learning Nanodegree Project Continuous Control: Training an agent with a double-jointed arm to reach a goal position.

## The Challenge
[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif
![Trained Agent][image1]

The challenge is to train an agent with a double-jointed arm to move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.



## Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name name python=3.6
	source activate name
	```
	- __Windows__: 
	```bash
	conda create --name name python=3.6 
	activate name
	```

2. Clone the repository and navigate to the folder.  Then, install the dependencies in the `requirements.txt` file.
```bash
git clone https://github.com/clauszitzelsberger/continuous-control-rl.git
cd continuous-control-rl
pip install pip install -r requirements.txt
```

3. Download the Unity Environment
Download the environment that matches your operation system, then place the file in the `Navigation_RL/` folder and unizip the file.  

	- **Version 1: 1 Agent:**  
		- [__Linux__](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
		- [__Mac OSX__](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
		- [__Windows (32-bit)__](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
		- [__Windows (64-bit)__](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)  
		- [__GPU version (e.g. for AWS)__](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip)  
	
	- **Version 2: 20 Agents:**  
		- [__Linux__](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
		- [__Mac OSX__](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
		- [__Windows (32-bit)__](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
		- [__Windows (64-bit)__](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)  
		- [__GPU version (e.g. for AWS)__](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip)

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `name` environment.  
```bash
python -m ipykernel install --user --name name --display-name "name"
```

5. Before running code in a notebook, change the kernel to match the `name` environment by using the drop-down `Kernel` menu. 
  
## Setup of repository

Apart from the `Readme.md` and the `requirements.txt` file this repository consists of the following files:

1. `agent.py`: Agent and ReplayBuffer classes with all required functions
2. `model.py`: Actor and Critc Network classes
3. `run.py`: Script which will train the agent. Can be run directly from the terminal.
4. `report.ipynb`: As an alternative to the `run.py` script this Jupyter Notebook has a step-by-step structure. Here the learning algorithm is described in detail
5. `checkpoint.pth`: Contains the weights of a successful QNetwork
