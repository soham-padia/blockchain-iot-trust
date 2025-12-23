import matplotlib.pyplot as plt
import numpy as np
import parameters as params
from simulation_manager import SimulationManager  # If you saved your SimulationManager class in a separate file; otherwise, import it from main.py
from plot_util import plot_cumulative_reward, plot_f1_score, plot_blockchain_length, plot_throughput

# Define the agent types you want to compare.
agent_types = ['rl', 'drl', 'marl']

# For a comparative run, we may use fewer episodes (adjust as needed).
num_episodes = 50

# Prepare a dictionary to store results.
results = {}

# Loop over agent types.
for agent in agent_types:
    print(f"\n=== Running simulation for agent: {agent.upper()} ===")
    # You can change the number of episodes per run.
    sim = SimulationManager(episodes=num_episodes, agent_type=agent)
    (final_report, predicted_malicious, trust_values, 
     episodes_list, cumulative_rewards, f1_scores, 
     blockchain_lengths, throughputs) = sim.run_simulation()
    
    results[agent] = {
        'episodes': episodes_list,
        'cumulative_rewards': cumulative_rewards,
        'f1_scores': f1_scores,
        'blockchain_lengths': blockchain_lengths,
        'throughputs': throughputs
    }

# --- Comparative Plots ---

# 1. Comparative Cumulative Reward Plot
plt.figure(figsize=(10, 6))
for agent in agent_types:
    plt.plot(results[agent]['episodes'], results[agent]['cumulative_rewards'],
             marker='o', label=agent.upper())
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title("Comparative Cumulative Reward vs. Episodes")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("comparative_cumulative_reward.png")
plt.show()

# 2. Comparative F1-Score Plot
plt.figure(figsize=(10, 6))
for agent in agent_types:
    plt.plot(results[agent]['episodes'], results[agent]['f1_scores'],
             marker='o', label=agent.upper())
plt.xlabel("Episode")
plt.ylabel("F1-Score")
plt.title("Comparative F1-Score vs. Episodes")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("comparative_f1_score.png")
plt.show()

# 3. Comparative Blockchain Length Plot
plt.figure(figsize=(10, 6))
for agent in agent_types:
    plt.plot(results[agent]['episodes'], results[agent]['blockchain_lengths'],
             marker='o', label=agent.upper())
plt.xlabel("Episode")
plt.ylabel("Blockchain Length (number of blocks)")
plt.title("Comparative Blockchain Growth vs. Episodes")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("comparative_blockchain_length.png")
plt.show()

# 4. Comparative Throughput Plot
plt.figure(figsize=(10, 6))
for agent in agent_types:
    plt.plot(results[agent]['episodes'], results[agent]['throughputs'],
             marker='o', label=agent.upper())
plt.xlabel("Episode")
plt.ylabel("Transaction Throughput (tx count in last block)")
plt.title("Comparative Throughput vs. Episodes")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("comparative_throughput.png")
plt.show()

# Optionally, you could also print a summary table.
print("\n=== Summary of Comparative Results ===")
for agent in agent_types:
    final_reward = results[agent]['cumulative_rewards'][-1]
    final_f1 = results[agent]['f1_scores'][-1]
    final_chain = results[agent]['blockchain_lengths'][-1]
    final_throughput = results[agent]['throughputs'][-1]
    print(f"{agent.upper()}: Final Cumulative Reward = {final_reward:.2f}, Final F1-Score = {final_f1:.4f}, "
          f"Chain Length = {final_chain}, Throughput = {final_throughput}")
