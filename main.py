from problem import Problem
from human import EpsilonGreedy, WSLS, ThompsonSampling, UCL, GittinsIndex

def main():
    print("------------Running main------------")
    print("Initial parameters:")

    n_arms = 4
    n_rounds = 1000
    human_policy = EpsilonGreedy
    print("n_arms:", n_arms)
    print("n_rounds:", n_rounds)
    print("human_policy:", human_policy)
    print()
    print("Creating MAB Problem instance")
    problem = Problem(n_arms, human_policy, n_rounds)
    print("Running game")
    problem.simulate()

if __name__ == "__main__":
    main()