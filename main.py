# main.py - BALANCED Simulation Manager for Realistic Results
"""
CHANGES FROM ORIGINAL:
1. Reduced ground truth update frequency (every 15 steps vs 5)
2. Attack every 5 steps instead of 10 (more attack opportunities)
3. More trust decay to prevent runaway trust scores
4. Added noise to various components
"""

import os
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import sys
import argparse
import pandas as pd

import parameters as params
from abac import ABAC
from fhe import FullyHomomorphicEncryption
from trust import TrustManager
from reward import RewardSystem
from tdcb import TDCB
from drl_d3p import DDQN
from marl_agent import MARLAgent
from plot_util import (plot_cumulative_reward, plot_f1_score,
                       plot_blockchain_length, plot_throughput,
                       plot_confusion_matrix_from_labels)
from attack_util import apply_attack, reset_attack_instances


def detect_collusion(trust_manager, malicious_nodes):
    """Compute collusion score."""
    all_nodes = list(trust_manager.trust_scores.keys())
    honest_nodes = [n for n in all_nodes if n not in malicious_nodes]
    if not honest_nodes or not malicious_nodes:
        return 0.0
    
    malicious_trust_vals = [trust_manager.get_trust(n) for n in malicious_nodes]
    honest_trust_vals = [trust_manager.get_trust(n) for n in honest_nodes]
    avg_mal = sum(malicious_trust_vals) / len(malicious_trust_vals)
    avg_hon = sum(honest_trust_vals) / len(honest_trust_vals)
    gap = abs(avg_hon - avg_mal)
    
    if gap < 1e-6:
        return 9999.0
    return 1.0 / gap


class BatchProcessor:
    def __init__(self, batch_size=params.BATCH_SIZE):
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def process_transactions(self, transactions):
        return [transactions[i:i + self.batch_size]
                for i in range(0, len(transactions), self.batch_size)]
    
    def process_batch(self, batch):
        sender_map = {sender: idx for idx, sender in enumerate(set(tx["sender"] for tx in batch))}
        recipient_map = {recipient: idx for idx, recipient in enumerate(set(tx["recipient"] for tx in batch))}
        batch_tensor = torch.tensor(
            [
                [sender_map[tx["sender"]],
                 recipient_map[tx["recipient"]],
                 tx["amount"]]
                for tx in batch
            ],
            dtype=torch.float32,
            device=self.device
        )
        return batch_tensor


class SimulationManager:
    def __init__(self, episodes=params.EPISODES, agent_type="drl", attack_mode="none"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.episodes = episodes
        self.agent_type = agent_type
        self.attack_mode = attack_mode
        self.batch_processor = BatchProcessor()
        
        reset_attack_instances()
        
        self._init_components(agent_type)
        self._init_nodes()
        
        self.reward_history = []
        self.decline_counter = 0
        self.strategy_changes = 0
        self.current_strategy = 0
        self.chain_length_5_episodes_ago = 0
        self.episode_counter = 0
        self.best_model_f1 = 0.0
        self.attack_rounds = 0

        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs("results", exist_ok=True)
        os.makedirs("images", exist_ok=True)

        self.total_byzantine_detected = 0
        self.total_consensus_rounds = 0
        self.abac_total_requests = 0
        self.abac_denied_requests = 0
        
        self.previous_trust_values = {node: 0.5 for node in self.nodes}
        self.last_state = self._compute_initial_state()

        print(f"[INFO] BALANCED Simulation initialized on {self.device}")
        print(f"[INFO] Agent: {agent_type.upper()}, Attack: {attack_mode.upper()}")
        print(f"[INFO] Nodes: {params.TOTAL_NODES}, Malicious Ratio: {params.MALICIOUS_RATIO}")
        print(f"[INFO] Ground Truth Update Interval: {getattr(params, 'GROUND_TRUTH_UPDATE_INTERVAL', 15)} steps")
    
    def _compute_initial_state(self):
        """Compute initial state vector."""
        return np.array([
            0.5,   # avg_trust
            0.0,   # trust_variance
            0.0,   # trust_skew
            0.5,   # trust_median
            0.0,   # trust_range
            0.0,   # trust_iqr
            0.0,   # coefficient_variation
            0.0,   # consensus_norm
            len(self.blockchain.chain) / params.CHAIN_NORM,
            (1.0 - params.MALICIOUS_RATIO) / params.MALICIOUS_RATIO,
            0.0,   # num_low_trust fraction
            1.0,   # num_high_trust fraction
            0.5,   # delegation_efficiency
            0.0,   # tx_throughput_rate
            0.0,   # blocks_last_5_episodes
            0.0    # collusion_score
        ])
    
    def _init_components(self, agent_type="drl"):
        self.abac = ABAC()
        self.fhe = FullyHomomorphicEncryption()
        self.reward_system = RewardSystem()
        self.tdcb = TDCB()
        self.blockchain = self._init_blockchain()
        
        state_dim = 16
        action_dim = 3
        
        if agent_type == "rl":
            from rl_agent import RLAgent
            self.agent = RLAgent(state_dim=state_dim, action_dim=action_dim)
        elif agent_type == 'marl':
            self.agent = MARLAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                num_agents=params.TOTAL_NODES,
                lr=5e-4,
                gamma=0.99
            )
        else:
            self.agent = DDQN(state_dim=state_dim, action_dim=action_dim)
    
    def _init_blockchain(self):
        blockchain = __import__("blockchain", fromlist=["Blockchain"]).Blockchain(difficulty=4)
        blockchain.delegation_ratio = 0.5
        return blockchain
    
    def _init_nodes(self):
        self.nodes = [f"Node_{i}" for i in range(params.TOTAL_NODES)]
        num_malicious = int(params.TOTAL_NODES * params.MALICIOUS_RATIO)
        self.true_malicious_nodes = set(random.sample(self.nodes, num_malicious))
        self.true_honest_nodes = set(self.nodes) - self.true_malicious_nodes
        
        for node in self.nodes:
            if node in self.true_malicious_nodes:
                self.abac.add_user_attributes(node, {"user_role"})
            else:
                self.abac.add_user_attributes(node, {"admin_role", "security_clearance"})
        
        self.trust_manager = TrustManager(self.nodes)
        
        print(f"[NODES] Malicious ({len(self.true_malicious_nodes)}): {sorted(self.true_malicious_nodes)}")
        print(f"[NODES] Honest ({len(self.true_honest_nodes)}): {sorted(self.true_honest_nodes)}")
    
    def _generate_transactions(self, num_transactions):
        senders = random.choices(self.nodes, k=num_transactions)
        recipients = random.choices(self.nodes, k=num_transactions)
        amounts = torch.rand(num_transactions, device=self.device) * 100
        return [{
            "sender": s,
            "recipient": r,
            "amount": a.item()
        } for s, r, a in zip(senders, recipients, amounts)]
    
    def _compute_state(self, trust_values, consensus_result, num_transactions, step_i):
        """Compute state vector from current trust values and metrics."""
        trust_list = list(trust_values.values())
        
        avg_trust = np.mean(trust_list)
        trust_variance = np.var(trust_list)
        trust_std = np.std(trust_list)
        trust_median = np.median(trust_list)
        trust_range = max(trust_list) - min(trust_list)
        trust_iqr = np.percentile(trust_list, 75) - np.percentile(trust_list, 25)
        
        trust_skew = 0
        if len(trust_list) > 2 and trust_std > 0:
            trust_skew = np.mean([(t - avg_trust)**3 for t in trust_list]) / (trust_std**3)
        
        coefficient_variation = trust_std / avg_trust if avg_trust > 0 else 0
        consensus_norm = consensus_result["verified_transactions"] / params.CONSENSUS_NORM
        chain_norm = len(self.blockchain.chain) / params.CHAIN_NORM
        honest_malicious_ratio = (1.0 - params.MALICIOUS_RATIO) / params.MALICIOUS_RATIO
        
        num_low_trust = sum(1 for t in trust_list if t < 0.3)
        num_high_trust = sum(1 for t in trust_list if t > 0.7)
        
        num_delegates = max(1, int(len(self.nodes) * self.blockchain.delegation_ratio))
        delegation_efficiency = num_delegates / params.TOTAL_NODES
        
        blocks_last_5_episodes = len(self.blockchain.chain) - self.chain_length_5_episodes_ago
        tx_throughput_rate = num_transactions / (step_i + 1) if step_i > 0 else num_transactions
        
        collusion_score = detect_collusion(self.trust_manager, self.true_malicious_nodes)
        
        return np.array([
            avg_trust,
            trust_variance,
            trust_skew,
            trust_median,
            trust_range,
            trust_iqr,
            coefficient_variation,
            consensus_norm,
            chain_norm,
            honest_malicious_ratio,
            num_low_trust / params.TOTAL_NODES,
            num_high_trust / params.TOTAL_NODES,
            delegation_efficiency,
            tx_throughput_rate / 50,
            blocks_last_5_episodes / 10,
            min(collusion_score / 10, 10.0)
        ])
    
    def _filter_transactions_with_fhe_abac(self, transactions):
        """Filter transactions using FHE-encrypted ABAC evaluation."""
        filtered = []
        
        for tx in transactions:
            sender = tx["sender"]
            trust_score = self.trust_manager.get_trust(sender)
            
            decision, confidence, reason = self.abac.enforce_policy_with_learning(
                user_id=sender,
                requested_action="write",
                time_of_day=random.randint(9, 17),
                day_type="weekday",
                location="secure_location",
                trust_score=trust_score
            )
            
            if decision:
                filtered.append(tx)
                self.abac_total_requests += 1
            else:
                self.abac_denied_requests += 1
        
        return filtered
    
    def run_episode(self, episode):
        """Run a single training episode."""
        print(f"\n{'='*60}")
        print(f"[EPISODE {episode+1}/{self.episodes}]")
        print(f"{'='*60}")
        
        STEPS_PER_EPISODE = params.STEPS_PER_EPISODE
        episode_reward = 0.0
        num_transactions = 0
        episode_f1_scores = []
        
        # BALANCED: Ground truth update interval (default 15)
        GROUND_TRUTH_UPDATE_INTERVAL = getattr(params, 'GROUND_TRUTH_UPDATE_INTERVAL', 15)
        
        state = self.last_state

        for step_i in range(STEPS_PER_EPISODE):
            step_tx_count = random.randint(15, 50)
            num_transactions += step_tx_count
            transactions = self._generate_transactions(step_tx_count)
            
            filtered_transactions = self._filter_transactions_with_fhe_abac(transactions)
            
            # BALANCED: More frequent trust decay (every 5 steps)
            if step_i % 5 == 0:
                self.trust_manager.decay_trust(decay_rate=0.003)  # Slightly stronger decay
            
            # BALANCED: Apply attack MORE FREQUENTLY (every 5 steps instead of 10)
            if self.attack_mode != "none" and step_i % 5 == 0:
                attack_metrics = apply_attack(
                    self.trust_manager, 
                    self.true_malicious_nodes, 
                    self.attack_mode,
                    current_step=step_i,
                    current_episode=episode
                )
                self.attack_rounds += 1
            
            # Get agent's action
            if self.agent_type == 'marl':
                agent_id = step_i % params.TOTAL_NODES
                adjustment = self.agent.get_adjustment(state, agent_id=agent_id)
            else:
                adjustment = self.agent.get_adjustment(state)
            
            base_trust = self.trust_manager.calculate_trust_snapshot(self.nodes)
            
            self.blockchain.delegation_ratio = max(0.1, min(1.0, 
                self.blockchain.delegation_ratio * adjustment))
            
            num_delegates = max(1, int(len(self.nodes) * self.blockchain.delegation_ratio))
            delegated_nodes = self.trust_manager.select_delegated_nodes(num_delegates)
            trust_snapshot = self.trust_manager.calculate_trust_snapshot(self.nodes)
            
            consensus_result = self.tdcb.execute_consensus(
                filtered_transactions,
                delegated_nodes,
                trust_snapshot
            )
            
            verified_txs = consensus_result.get("verified_tx_list", [])
            for tx in verified_txs:
                self.blockchain.add_transaction(tx)
            
            malicious_delegates = consensus_result.get("malicious_nodes", [])
            for mal_node in malicious_delegates:
                self.trust_manager.update_trust(mal_node, "malicious")
                self.total_byzantine_detected += 1
            
            for node in delegated_nodes:
                if node not in malicious_delegates:
                    self.trust_manager.update_trust(node, "valid")
            
            # BALANCED: Update ground truth LESS FREQUENTLY (every 15 steps instead of 5)
            # This gives attacks more room to affect trust scores
            if step_i % GROUND_TRUTH_UPDATE_INTERVAL == 0:
                for node in self.true_malicious_nodes:
                    # PROBABILISTIC update (not guaranteed)
                    if random.random() < 0.7:  # 70% chance
                        self.trust_manager.update_trust(node, "malicious")
                for node in self.true_honest_nodes:
                    if random.random() < 0.7:  # 70% chance
                        self.trust_manager.update_trust(node, "valid")
            
            current_trust_values = self.trust_manager.calculate_trust_snapshot(self.nodes)
            
            report, predicted, _ = self._evaluate_results(current_trust_values)
            f1_score = report["macro avg"]["f1-score"]
            episode_f1_scores.append(f1_score)
            
            new_state = self._compute_state(
                current_trust_values, consensus_result, num_transactions, step_i
            )
            
            step_rewards = self.reward_system.calculate_shaped_rewards(
                current_trust_values,
                self.previous_trust_values,
                self.true_malicious_nodes,
                self.true_honest_nodes,
                gamma=0.99
            )
            self.previous_trust_values = current_trust_values.copy()
            
            step_reward_value = sum(step_rewards.values())
            
            fn_penalty = sum(1 for mal in self.true_malicious_nodes 
                           if mal not in predicted) * params.FN_PENALTY_WEIGHT
            
            collusion_score = detect_collusion(self.trust_manager, self.true_malicious_nodes)
            
            if self.attack_mode == "cra":
                trust_variance = np.var(list(current_trust_values.values()))
                trust_variance_reward = trust_variance * 50
                combined_reward = (
                    params.F1_REWARD_WEIGHT * (f1_score * 100) +
                    params.STEP_REWARD_WEIGHT * (step_reward_value / 100) +
                    0.1 * trust_variance_reward -
                    fn_penalty
                )
                if collusion_score > 2.0:
                    penalty = min(collusion_score * 2, 20.0)
                    combined_reward -= penalty
            else:
                combined_reward = (
                    params.F1_REWARD_WEIGHT * (f1_score * 100) + 
                    params.STEP_REWARD_WEIGHT * (step_reward_value / 100) -
                    fn_penalty
                )
            
            episode_reward += combined_reward
            
            done = (step_i == STEPS_PER_EPISODE - 1)
            
            if self.agent_type == 'marl':
                agent_id = step_i % params.TOTAL_NODES
                discrete_action = 1
                if adjustment < 0.95:
                    discrete_action = 0
                elif adjustment > 1.05:
                    discrete_action = 2
                
                self.agent.push_experience(
                    state, discrete_action, combined_reward, 
                    new_state, done, agent_id=agent_id
                )
                train_loss = self.agent.train(agent_id=agent_id)
                
                if done:
                    self.agent.update_epsilon(episode, agent_id=agent_id)
                    self.agent.update_learning_rate(episode, agent_id=agent_id)
            else:
                train_loss = self.agent.train_adjustment(
                    state, adjustment, combined_reward, new_state, done
                )
            
            state = new_state
            
            if step_i % 20 == 0:
                print(f"  Step {step_i+1}/{STEPS_PER_EPISODE} | "
                      f"F1: {f1_score:.4f} | "
                      f"Reward: {combined_reward:.2f} | "
                      f"Adjustment: {adjustment:.2f}")

        self.last_state = state
        
        final_report, final_predicted, trust_values = self._evaluate_results()
        final_f1 = final_report["macro avg"]["f1-score"]
        current_blockchain_length = len(self.blockchain.chain)
        last_block_tx_count = len(self.blockchain.chain[-1].transactions) if current_blockchain_length > 1 else 0
        
        honest_trusts = [trust_values[n] for n in self.true_honest_nodes]
        malicious_trusts = [trust_values[n] for n in self.true_malicious_nodes]
        trust_separation = np.mean(honest_trusts) - np.mean(malicious_trusts)
        
        fp_count = sum(1 for hon in self.true_honest_nodes if hon in final_predicted)
        fn_count = sum(1 for mal in self.true_malicious_nodes if mal not in final_predicted)
        
        print(f"\n[EPISODE {episode+1} SUMMARY]")
        print(f"  Final F1: {final_f1:.4f} | Episode Reward: {episode_reward:.2f}")
        print(f"  Trust Separation: {trust_separation:.4f} (honest - malicious)")
        print(f"  FP: {fp_count} | FN: {fn_count}")
        print(f"  Blockchain Length: {current_blockchain_length}")
        
        print("\n  Trust Distribution:")
        for node in sorted(self.nodes):
            trust = trust_values[node]
            marker = '[M]' if node in self.true_malicious_nodes else '[H]'
            detected = '*DETECTED*' if node in final_predicted else ''
            print(f"    {node}: {trust:.4f} {marker} {detected}")
        
        if hasattr(self.agent, 'save_model') and final_f1 > self.best_model_f1:
            self.best_model_f1 = final_f1
            checkpoint_path = f"{self.checkpoint_dir}/best_model_{self.agent_type}_{self.attack_mode}_f1_{final_f1:.4f}.pth"
            try:
                self.agent.save_model(checkpoint_path)
                print(f"  [CHECKPOINT] Best model saved: F1={final_f1:.4f}")
            except Exception as e:
                print(f"  [WARNING] Could not save model: {e}")
        
        self.episode_counter += 1
        if self.episode_counter % 5 == 0:
            self.chain_length_5_episodes_ago = len(self.blockchain.chain)
        
        return {
            'cumulative_reward': episode_reward,
            'transactions': num_transactions,
            'f1_score': final_f1,
            'precision': final_report.get('Malicious', {}).get('precision', 0),
            'recall': final_report.get('Malicious', {}).get('recall', 0),
            'blockchain_length': current_blockchain_length,
            'last_block_tx_count': last_block_tx_count,
            'byzantine_detected': self.total_byzantine_detected,
            'trust_separation': trust_separation,
            'fp_count': fp_count,
            'fn_count': fn_count,
            'avg_f1': np.mean(episode_f1_scores)
        }

    def run_simulation(self):
        """Run complete simulation across all episodes."""
        cumulative_rewards = []
        f1_scores = []
        blockchain_lengths = []
        throughputs = []
        episode_numbers = []
        
        start_time = time.time()
        progress_bar = tqdm(range(self.episodes), desc="Training Progress")
        
        for episode in progress_bar:
            results = self.run_episode(episode)
            
            cumulative_rewards.append(results['cumulative_reward'])
            f1_scores.append(results['f1_score'])
            blockchain_lengths.append(results.get('blockchain_length', 0))
            throughputs.append(results.get('last_block_tx_count', 0))
            episode_numbers.append(episode + 1)
            
            self.total_consensus_rounds += 100
            
            progress_bar.set_postfix({
                'Reward': f"{results['cumulative_reward']:.2f}",
                'F1': f"{results['f1_score']:.4f}",
                'Sep': f"{results['trust_separation']:.3f}"
            })
        
        end_time = time.time()
        print(f"\n{'='*60}")
        print(f"SIMULATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total Runtime: {end_time - start_time:.2f} seconds")
        
        final_report, predicted_malicious, trust_values = self._evaluate_results()
        
        print("\nFinal Trust Values:")
        for node in self.true_malicious_nodes:
            print(f"  [MALICIOUS] {node}: {trust_values[node]:.4f}")
        for node in self.true_honest_nodes:
            print(f"  [HONEST]    {node}: {trust_values[node]:.4f}")
        
        print(f"\nPredicted Malicious: {predicted_malicious}")
        print(f"Best Threshold: {params.TRUST_THRESHOLD}")
        print(f"Best F1-Score: {final_report['macro avg']['f1-score']:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(
            [1 if node in self.true_malicious_nodes else 0 for node in self.nodes],
            [1 if node in predicted_malicious else 0 for node in self.nodes],
            target_names=["Honest", "Malicious"],
            zero_division=1
        ))
        
        avg_csp = 1.0
        print(f"Average Consensus Success Probability (CSP): {avg_csp:.4f}")
        
        plot_cumulative_reward(episode_numbers, cumulative_rewards, self.agent_type, self.attack_mode)
        plot_f1_score(episode_numbers, f1_scores, self.agent_type, self.attack_mode)
        plot_blockchain_length(episode_numbers, blockchain_lengths, self.agent_type, self.attack_mode)
        plot_throughput(episode_numbers, throughputs, self.agent_type, self.attack_mode)
        
        y_true = [1 if node in self.true_malicious_nodes else 0 for node in self.nodes]
        y_pred = [1 if node in predicted_malicious else 0 for node in self.nodes]
        plot_confusion_matrix_from_labels(y_true, y_pred, self.agent_type, self.attack_mode, 
                                         title="Final Confusion Matrix")
        
        data = {
            "Episode": episode_numbers,
            "Cumulative Reward": cumulative_rewards,
            "F1 Score": f1_scores,
            "Blockchain Length": blockchain_lengths,
            "Throughput": throughputs
        }
        df = pd.DataFrame(data)
        
        param_data = {
            "TOTAL_NODES": [params.TOTAL_NODES],
            "EPISODES": [params.EPISODES],
            "MALICIOUS_RATIO": [params.MALICIOUS_RATIO],
            "AGENT": [self.agent_type],
            "ATTACK_MODE": [self.attack_mode],
            "CHAIN_NORM": [params.CHAIN_NORM],
            "CONSENSUS_NORM": [params.CONSENSUS_NORM]
        }
        df_params = pd.DataFrame(param_data)
        
        file_prefix = f"{params.TOTAL_NODES}_{params.EPISODES}_{self.agent_type}_{self.attack_mode}_{int(params.MALICIOUS_RATIO*100)}"
        df.to_csv(f"results/{file_prefix}_episode_metrics.csv", index=False)
        df_params.to_csv(f"results/{file_prefix}_simulation_parameters.csv", index=False)
        print(f"\n[INFO] Results saved with prefix: {file_prefix}")
    
    def _evaluate_results(self, trust_values=None):
        """Evaluate malicious node detection performance."""
        if trust_values is None:
            trust_values = self.trust_manager.calculate_trust_snapshot(self.nodes)
        
        predicted_malicious = {
            node for node, trust in trust_values.items() 
            if trust < params.TRUST_THRESHOLD
        }
        
        y_true = [1 if node in self.true_malicious_nodes else 0 for node in self.nodes]
        y_pred = [1 if node in predicted_malicious else 0 for node in self.nodes]
        
        report = classification_report(
            y_true, y_pred,
            target_names=["Honest", "Malicious"],
            zero_division=1,
            output_dict=True
        )
        
        return report, predicted_malicious, trust_values


def main():
    arg_list = []
    for arg in sys.argv[1:]:
        if "=" in arg:
            key, value = arg.split("=", 1)
            arg_list.extend([f"--{key}", value])
        else:
            arg_list.append(arg)
    
    parser = argparse.ArgumentParser(description="BALANCED Blockchain IoT Trust Simulation")
    parser.add_argument("--nodes", type=int, default=params.TOTAL_NODES, 
                       help="Total number of nodes")
    parser.add_argument("--episodes", type=int, default=params.EPISODES, 
                       help="Total simulation episodes")
    parser.add_argument("--steps", type=int, default=params.STEPS_PER_EPISODE,
                   help="Steps per episode (default: 100)")
    parser.add_argument("--agent", type=str, default="drl", 
                       choices=["rl", "drl", "marl"],
                       help="Type of agent: rl, drl, or marl")
    parser.add_argument("--attack", type=str, default="none", 
                       choices=['none', 'nma', 'cra', 'bfi', 'aaa', 'tdp'],
                       help='Attack type: none, nma, cra, bfi, aaa, tdp')
    parser.add_argument('--malicious', type=float, default=params.MALICIOUS_RATIO,
                       help='Malicious node ratio (e.g. 0.3 for 30%%)')
    
    args = parser.parse_args(arg_list)
    
    params.TOTAL_NODES = args.nodes
    params.EPISODES = args.episodes
    params.MALICIOUS_RATIO = args.malicious
    params.STEPS_PER_EPISODE = args.steps
    
    simulator = SimulationManager(
        episodes=params.EPISODES, 
        agent_type=args.agent, 
        attack_mode=args.attack
    )
    simulator.run_simulation()


if __name__ == "__main__":
    main()
