# reward.py
from typing import Dict
import torch
import numpy as np

# Auto-select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RewardSystem:
    def __init__(self):
        pass

    def calculate_rewards(self, trust_values: Dict[str, float],
                        true_malicious_nodes=None, true_honest_nodes=None) -> Dict[str, float]:
        """
        Industry-standard multi-objective reward function.
        Optimized for F1-score maximization.
        """
        trust_tensor = torch.tensor(list(trust_values.values()), dtype=torch.float32, device=device)
        
        # Component 1: Trust-based baseline (encourage differentiation)
        # Reward is ZERO at trust=0.3, negative below, positive above
        baseline_rewards = (trust_tensor - 0.3) * 30
        rewards = {node: reward.item() for node, reward in zip(trust_values.keys(), baseline_rewards)}
        
        # Component 2: Detection accuracy bonuses (if ground truth available)
        if true_malicious_nodes and true_honest_nodes:
            honest_trusts = [trust_values[n] for n in true_honest_nodes]
            malicious_trusts = [trust_values[n] for n in true_malicious_nodes]
            
            # Calculate trust separation
            separation = np.mean(honest_trusts) - np.mean(malicious_trusts)
            separation_bonus = separation * 100  # Strong signal for separation
            
            for node in trust_values.keys():
                node_trust = trust_values[node]
                
                if node in true_malicious_nodes:
                    # Reward for correctly identifying malicious (low trust)
                    detection_reward = (1.0 - node_trust) * 50
                    rewards[node] += detection_reward
                else:
                    # Reward for correctly trusting honest (high trust)
                    maintenance_reward = node_trust * 40
                    rewards[node] += maintenance_reward
                
                # Add separation bonus to all
                rewards[node] += separation_bonus / len(trust_values)
        
        # Component 3: Variance bonus (encourage exploration)
        trust_variance = torch.var(trust_tensor).item()
        variance_bonus = trust_variance * 50
        
        for node in rewards:
            rewards[node] += variance_bonus / len(rewards)
        
        return rewards

    def calculate_shaped_rewards(
        self,
        trust_values: Dict[str, float],
        previous_trust: Dict[str, float],
        true_malicious_nodes: set,
        true_honest_nodes: set,
        gamma: float = 0.99
    ) -> Dict[str, float]:
        """
        Calculate rewards based on trust values with improved incentives to detect collusive patterns.
        
        Args:
            trust_values: Dictionary of node trust values
            true_malicious_nodes: Set of actual malicious nodes (optional for validation)
            true_honest_nodes: Set of actual honest nodes (optional for validation)
        
        Multi-objective reward function with security, accuracy, and efficiency components.
        Reward components:
        1. Detection accuracy (F1, precision, recall)
        2. Trust distribution health
        3. False positive/negative penalties
        4. Attack resistance bonus
        """
        # print("Calculating rewards...")s
        
         # Calculate trust separation potential
        def potential(trust_dict):
            honest_avg = np.mean([trust_dict.get(n, 0.5) for n in true_honest_nodes])
            malicious_avg = np.mean([trust_dict.get(n, 0.5) for n in true_malicious_nodes])
            return (honest_avg - malicious_avg) * 100  # Scale for significance
        
        current_potential = potential(trust_values)
        previous_potential = potential(previous_trust)
        
        # Shaping bonus encourages increasing trust separation
        shaping_bonus = gamma * current_potential - previous_potential
        
        # Base rewards (existing logic)
        base_rewards = self.calculate_rewards(trust_values, true_malicious_nodes, true_honest_nodes)
        
        # Add potential-based shaping to each node
        shaped_rewards = {}
        for node in trust_values:
            shaped_rewards[node] = base_rewards[node] + (shaping_bonus / len(trust_values))
        
        return shaped_rewards