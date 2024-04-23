# An Extension To The Assistive Multi-Armed Bandit

## Overview
In this final project, I explore the "Assistive Multi-Armed Bandit" framework where a robot assists a human in a bandit task to enhance preference learning under uncertainty. 
This research is highly relevant to our course on Multi-Armed Bandits and Regret Minimization, as it integrates these concepts to optimize human-robot interaction dynamics and thus minimize the human regret. 
The project entails a thorough summary of the paper's models and findings, implementation of the proposed algorithm, and a proof sketch of key theoretical claims. 
Further, I propose an extension that adapts the model to dynamically changing arms. 
This extension aims to generalize the framework, offering insights into more complex, real-world scenarios where learning conditions are not static and the set of available actions in each time point is not the same (mortal arms).
This project implements a simulation of multi-armed bandit (MAB) scenarios using different policies and strategies, including human-like decision making and robot interaction through reinforcement learning techniques like PPO (Proximal Policy Optimization). The system models a decision-making environment where both a human-like agent and an automated robot interact to optimize their choices among multiple options with uncertain rewards.

## Structure
The project is organized into several Python modules, each serving a specific function in the Assistive MAB simulation:

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
1. Ensure you have Python and necessary libraries installed (`numpy`, `torch`).
2. Navigate to the project directory in your terminal.
3. Run the command: `python main.py`

## Dependencies
- Python 3.10
- NumPy
- PyTorch
## Install Dependencies:
- You can use the `requirements.txt` file to install dependencies:
```bash
pip install -r requirements.txt
```
## Extension: The Mortal Assistive MAB
The Mortal Assistive Multi-Armed Bandit problem combines the challenges of adaptive decision-making with the reality that options may have limited lifetimes. This reflects real-world scenarios where products might go out of stock, promotional campaigns end, or user interests shift, necessitating continual re-evaluation and exploration of the available choices.
The motivation for integrating the assistive aspect into the mortal MAB framework is to enhance systems' abilities to adaptively interact with users or environments. For instance, in personalized advertising, an ad that is effective for a certain audience may only be relevant during a specific event or within a limited time window due to budget constraints or campaign durations. Similarly, in stock trading, certain investment opportunities might have short-lived viability, requiring a strategy that not only learns from past interactions but also anticipates and reacts to expiring options effectively.
The adapted ADAPTIVEGREEDY algorithm for the Mortal Assistive MAB problem combines the ideas of the adaptive greedy heuristic with the PPO algorithm. 

## Main Modifications
The `select_arm` method now incorporates the adaptive greedy heuristic. With probability epsilon, the robot explores by selecting a random active arm. With probability 1 - epsilon, the robot exploits by selecting the arm with the highest Upper Confidence Bound (UCB) value among the active arms. The UCB value for each arm is calculated based on the average reward and the number of times the arm has been pulled.
The `update_arm_stats` method is introduced to update the count and total reward for each arm after it is pulled. This information is used to calculate the UCB values.
The prepare_input_sequence method remains the same as before, which prepares the input sequence for the RNN based on the human and robot choice history.
The `update_human_choice` and `update_actual_pulls` methods are used to update the human observations and robot pulls, respectively.

The adapted algorithm addresses the challenges of the Mortal Assistive MAB problem by:

Considering only the active arms during the arm selection process. This ensures that the robot does not select expired arms.
Using the adaptive greedy heuristic to balance exploration and exploitation. The robot explores with a small probability epsilon to discover potentially better arms, while exploiting the best arm based on the UCB values to maximize rewards.
Updating the arm statistics (count and total reward) after each pull, which allows the algorithm to adapt to the changing dynamics of the mortal bandit setting.

By incorporating these modifications, the adapted ADAPTIVEGREEDY algorithm can effectively handle the Mortal Assistive MAB problem, learning from both the human choices and the rewards obtained by the robot while considering the limited lifetimes of the arms.


## Haven't Finished yet

### `mortal_bandit.py`

- **Purpose**: The `MortalBandit` class is designed to encapsulate the mortal bandit model where each arm has a finite lifetime. It handles the generation of rewards and monitors the lifespan of each arm, simulating their expiration over time.

- **Usage**: This class is instantiated by `mortal_problem.py` to simulate the mortal aspect of the MAB during gameplay. It determines the rewards based on both the probability distribution of the arms and their respective lifetimes.

### `mortal_human_policy.py`

- **Purpose**: 
This module contains classes that extend the decision-making policies for a human interacting with a mortal bandit. Classes such as `MortalEpsilonGreedy` take into account not only the past rewards but also the remaining lifetime of each arm.

- **Usage**: 
`mortal_problem.py` uses these policies to simulate human strategies in the face of a bandit with arms that may expire, affecting the decision-making process.

### `mortal_robot.py`

- **Purpose**: 
`MortalRobot` applies decision-making strategies suitable for the mortal bandit context. It uses the Proximal Policy Optimization (PPO) algorithm alongside heuristics like Upper Confidence Bound (UCB) that are adapted to only consider active arms.

- **Usage**: 
This module's logic is utilized by `mortal_problem.py` to determine the robot's actions in the simulation. It ensures that the robot's strategy adapts to the changing availability of arms.

### `mortal_problem.py`

- **Purpose**: 
Serves as the orchestrator for the entire simulation. It integrates the mortal bandit, human policies, and robot's decision-making process into a cohesive simulation environment that accounts for the mortality of the arms.

- **Usage**: 
Executed by `main.py` to initiate and control the flow of the Mortal Assistive MAB simulation, managing the rounds and interactions between the human, robot, and bandit.

