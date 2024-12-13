from car_rental.environment import CarRentalEnv
from car_rental.helpers import save_policy_plot


if __name__ == "__main__":
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Run car rental problem with specified algorithm."
    )
    parser.add_argument(
        "algorithm",
        choices=["policy_iteration", "value_iteration", "q_learning"],
        help="Algorithm to use for training the agent.",
    )
    args = parser.parse_args()

    # Route to the particular file based on the algorithm
    if args.algorithm == "policy_iteration":
        from car_rental.algorithms.policy_iteration import PolicyIterationAgent as Agent
    elif args.algorithm == "value_iteration":
        from car_rental.algorithms.value_iteration import ValueIterationAgent as Agent
    elif args.algorithm == "q_learning":
        from car_rental.algorithms.q_learning import QLearningAgent as Agent

    # Initialize the environment
    env = CarRentalEnv()

    # Initialize the RL agent
    agent = Agent(env)

    # Train the agent (policy iteration, value iteration, etc.)
    agent.train(max_iterations=1000, threshold=0.01)

    # Test the policy
    test_state = (10, 10)  # Example state
    best_action = agent.act(test_state)
    print(f"Best action for state {test_state}: {best_action}")

    # Save the policy
    agent.save_policy(f"{args.algorithm}_policy.npy")
    save_policy_plot(agent.policy, f"{args.algorithm}_policy.png")
