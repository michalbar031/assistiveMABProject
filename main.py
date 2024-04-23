from problem import Problem
from human import EpsilonGreedy, WSLS, ThompsonSampling, UCL, GittinsIndex

def main():
    n_arms = 4
    n_rounds = 1000
    human_policy = EpsilonGreedy

    problem = Problem(n_arms, human_policy, n_rounds)
    problem.simulate()

if __name__ == "__main__":
    main()