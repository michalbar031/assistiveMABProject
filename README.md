# An Extension To The Assistive Multi-Armed Bandit

## Overview
In this final project, I explore the "Assistive Multi-Armed Bandit" framework where a robot assists a human in a bandit task to enhance preference learning under uncertainty. 
This research is highly relevant to our course on Multi-Armed Bandits and Regret Minimization, as it integrates these concepts to optimize human-robot interaction dynamics and thus minimize the human regret . 
The project entails a thorough summary of the paper's models and findings, implementation of the proposed algorithm, and a proof sketch of key theoretical claims. 
Further, I propose an extension that adapts the model to dynamically changing arms. 
This extension aims to generalize the framework, offering insights into more complex, real-world scenarios where learning conditions are not static and the set of available actions in each time point is not the same (mortal arms).
This project implements a simulation of multi-armed bandit (MAB) scenarios using different policies and strategies, including human-like decision making and robot interaction through reinforcement learning techniques like PPO (Proximal Policy Optimization). The system models a decision-making environment where both a human-like agent and an automated robot interact to optimize their choices among multiple options with uncertain rewards.

## Structure
The project is organized into several Python modules, each serving a specific function in the MAB simulation:

### `main.py`
- **Purpose**: Serves as the entry point of the program. It initializes the simulation environment, including the number of arms, rounds, and the type of human policy to be simulated.
- **How to Run**: Execute `python main.py` to start the simulation. The script will automatically invoke other modules as needed to run the defined simulation.

### `bandit.py`
- **Purpose**: Defines different types of bandit models, such as `BernoulliBandit` and `BetaBernoulliBandit`. These classes are responsible for the stochastic behavior of each arm in the bandit setup, providing rewards based on predefined probabilities.
- **Usage**: Instantiated by `problem.py` to manage the reward mechanism during the simulation.

### `human.py`
- **Purpose**: Contains various classes that define different human decision-making policies, such as `EpsilonGreedy`, `WSLS` (Win-Stay, Lose-Shift), and others. These classes simulate how a human might choose an arm based on past experiences.
- **Usage**: Used by `problem.py` to simulate human interaction with the bandit.

### `robot.py`
- **Purpose**: Implements the robot's decision-making logic using a neural network and the PPO algorithm to optimize its choices against the human player and the bandit.
- **Usage**: Interacts within the `problem.py` environment to make decisions based on the trained model.

### `ppo.py`
- **Purpose**: Implements the Proximal Policy Optimization algorithm, which is used to train the robot's decision-making model based on the rewards received from the bandit.
- **Usage**: Called by `robot.py` for training the decision-making model.

### `problem.py`
- **Purpose**: Orchestrates the simulation by combining the bandit, human, and robot modules. It manages the sequence of events in each round of the simulation.
- **Usage**: Called directly from `main.py` to start and manage the MAB simulation.

### `rnn.py`
- **Purpose**: Defines a recurrent neural network model used by the robot to make decisions.
- **Usage**: Imported by `robot.py` to build the decision-making model.

## Running the Simulation
1. Ensure you have Python and necessary libraries installed (`numpy`, `torch`, etc.).
2. Navigate to the project directory in your terminal.
3. Run the command: `python main.py`
4. Follow any on-screen prompts (if implemented) or check the terminal output to review the results of the simulation.

## Dependencies
- Python 3.10
- NumPy
- PyTorch


