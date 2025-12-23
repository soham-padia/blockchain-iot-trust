# trust.py - BALANCED Trust Manager with Realistic Detection Difficulty
"""
Trust Manager tuned for realistic detection scenarios:
- Weaker update magnitudes
- Higher noise/variance
- Slower convergence
- Attacks can meaningfully affect trust scores
"""

import torch
import numpy as np
import heapq
from scipy.stats import beta
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrustManager:
    """
    Bayesian Trust Manager using Beta distribution for trust estimation.
    
    BALANCED VERSION:
    - Reduced update magnitudes (attacks can compete)
    - Higher initial variance (uncertain detection)
    - Noise in updates (realistic uncertainty)
    """
    def __init__(self, nodes, config=None):
        """
        Initialize TrustManager with balanced priors.
        """
        # BALANCED configuration - weaker updates, more uncertainty
        self.config = config or {
            'initial_alpha': 8.0,        # REDUCED from 10.0
            'initial_beta': 8.0,         # REDUCED from 10.0
            'valid_boost': 3.0,          # REDUCED from 8.0
            'invalid_penalty': 2.0,      # REDUCED from 3.0
            'malicious_penalty': 4.0,    # REDUCED from 10.0
            'malicious_alpha_decay': 0.92,  # INCREASED from 0.8 (less aggressive)
            'decay_rate': 0.005,         # INCREASED from 0.001
            'min_trust': 0.01,
            'max_trust': 0.99,
            'min_alpha': 1.0,
            'min_beta': 1.0,
            'noise_level': 0.08,         # NEW: Add noise to updates
            'attack_resistance': 0.7,    # NEW: How much attack effects are scaled
        }
        
        self.alpha = {}
        self.beta_param = {}
        self.trust_scores = {}
        self.update_counts = {node: 0 for node in nodes}
        self.uncertainty = {}

        # INCREASED randomness in initial trust - makes detection harder
        base_alpha = self.config['initial_alpha']
        base_beta = self.config['initial_beta']
        
        for node in nodes:
            # Add MORE randomness to initial trust (±0.20 from 0.5)
            noise = random.uniform(-4, 4)  # INCREASED from ±3
            self.alpha[node] = base_alpha + noise
            self.beta_param[node] = base_beta - noise
            
            # Add additional per-node variance
            extra_variance = random.uniform(-2, 2)
            self.alpha[node] += extra_variance
            self.beta_param[node] += abs(extra_variance) * 0.5
            
            total = self.alpha[node] + self.beta_param[node]
            self.trust_scores[node] = self.alpha[node] / total
            self.uncertainty[node] = self._calculate_uncertainty(node)

        print(f"[TRUST] Initialized {len(nodes)} nodes with HIGH-VARIANCE Bayesian trust")
        print(f"[TRUST] Update strengths: valid={self.config['valid_boost']}, "
              f"malicious={self.config['malicious_penalty']}, "
              f"noise={self.config['noise_level']}")
    
    def _calculate_uncertainty(self, node_id):
        """Calculate Beta distribution variance as uncertainty measure."""
        a = self.alpha.get(node_id, self.config['initial_alpha'])
        b = self.beta_param.get(node_id, self.config['initial_beta'])
        variance = (a * b) / ((a + b)**2 * (a + b + 1))
        return variance
    
    def _recalculate_trust(self, node_id):
        """Recalculate trust score from alpha/beta parameters."""
        a = max(self.config['min_alpha'], self.alpha[node_id])
        b = max(self.config['min_beta'], self.beta_param[node_id])
        
        self.alpha[node_id] = a
        self.beta_param[node_id] = b
        
        total = a + b
        self.trust_scores[node_id] = a / total
        self.trust_scores[node_id] = np.clip(
            self.trust_scores[node_id],
            self.config['min_trust'],
            self.config['max_trust']
        )
        
        self.uncertainty[node_id] = self._calculate_uncertainty(node_id)
    
    def update_trust(self, node_id, outcome):
        """
        Update trust based on observed outcome.
        
        BALANCED VERSION:
        - Weaker update magnitudes
        - Added noise to updates
        - Probabilistic application (not guaranteed)
        """
        if node_id not in self.alpha:
            return
        
        self.update_counts[node_id] += 1
        
        # Add noise to updates (realistic uncertainty)
        noise = random.gauss(0, self.config['noise_level'])
        
        if outcome == "valid":
            # Positive evidence with noise
            boost = self.config['valid_boost'] + noise
            boost = max(0.5, boost)  # Ensure positive but can be reduced
            self.alpha[node_id] += boost
            
        elif outcome == "invalid":
            # Negative evidence with noise
            penalty = self.config['invalid_penalty'] + noise
            penalty = max(0.5, penalty)
            self.beta_param[node_id] += penalty
            
        elif outcome == "malicious":
            # Strong negative evidence with noise
            penalty = self.config['malicious_penalty'] + noise
            penalty = max(1.0, penalty)
            self.beta_param[node_id] += penalty
            
            # Decay alpha (with noise in decay rate)
            decay = self.config['malicious_alpha_decay'] + random.gauss(0, 0.02)
            decay = np.clip(decay, 0.85, 0.98)
            self.alpha[node_id] *= decay
        
        self._recalculate_trust(node_id)
    
    def adjust_trust(self, node_id, amount):
        """
        Direct trust adjustment (for attack simulations).
        
        BALANCED VERSION:
        - Stronger effect (attacks are more impactful)
        - Less resistance to manipulation
        """
        if node_id not in self.trust_scores:
            return
        
        # INCREASED attack effectiveness (was * 10, now * 15)
        scale_factor = 15.0
        
        # Add small noise to make attacks less predictable
        noise = random.gauss(0, 0.5)
        
        if amount > 0:
            adjustment = (amount * scale_factor) + noise
            self.alpha[node_id] += max(0, adjustment)
        else:
            adjustment = (abs(amount) * scale_factor) + noise
            self.beta_param[node_id] += max(0, adjustment)
        
        self._recalculate_trust(node_id)
    
    def decay_trust(self, decay_rate=None):
        """
        Apply gradual trust decay to all nodes.
        
        BALANCED: Faster decay = trust scores regress to 0.5 more
        """
        if decay_rate is None:
            decay_rate = self.config['decay_rate']
        
        if decay_rate <= 0:
            return
        
        for node_id in self.alpha:
            # Decay both alpha and beta
            self.alpha[node_id] *= (1.0 - decay_rate)
            self.beta_param[node_id] *= (1.0 - decay_rate)
            
            # Add small noise during decay
            noise = random.gauss(0, 0.1)
            self.alpha[node_id] += noise
            self.beta_param[node_id] -= noise * 0.5
            
            # Ensure minimums
            self.alpha[node_id] = max(self.config['min_alpha'], self.alpha[node_id])
            self.beta_param[node_id] = max(self.config['min_beta'], self.beta_param[node_id])
            
            self._recalculate_trust(node_id)
    
    def get_trust(self, node_id):
        """Get current trust score for a node."""
        return self.trust_scores.get(node_id, 0.5)
    
    def get_trust_with_uncertainty(self, node_id):
        """Get trust score with confidence interval."""
        a = max(self.config['min_alpha'], self.alpha.get(node_id, self.config['initial_alpha']))
        b = max(self.config['min_beta'], self.beta_param.get(node_id, self.config['initial_beta']))
        
        try:
            lower = beta.ppf(0.025, a, b)
            upper = beta.ppf(0.975, a, b)
        except:
            lower, upper = 0.0, 1.0
        
        return {
            'trust': self.trust_scores.get(node_id, 0.5),
            'lower_bound': lower,
            'upper_bound': upper,
            'uncertainty': self.uncertainty.get(node_id, 0.1)
        }
    
    def select_delegated_nodes(self, num_delegates):
        """Select delegates using Thompson Sampling."""
        samples = {}
        for node_id in self.trust_scores:
            a = max(self.config['min_alpha'], self.alpha[node_id])
            b = max(self.config['min_beta'], self.beta_param[node_id])
            samples[node_id] = np.random.beta(a, b)
        
        top_nodes = heapq.nlargest(
            num_delegates,
            samples.items(),
            key=lambda item: item[1]
        )
        return [node_id for node_id, _ in top_nodes]
    
    def select_delegates_ucb(self, num_delegates, exploration_weight=1.0):
        """Select delegates using Upper Confidence Bound (UCB)."""
        ucb_scores = {}
        for node_id in self.trust_scores:
            trust = self.trust_scores[node_id]
            uncertainty = self.uncertainty.get(node_id, 0.1)
            ucb_scores[node_id] = trust + exploration_weight * np.sqrt(uncertainty)
        
        top_nodes = heapq.nlargest(
            num_delegates,
            ucb_scores.items(),
            key=lambda item: item[1]
        )
        return [node_id for node_id, _ in top_nodes]
    
    def calculate_trust_snapshot(self, nodes):
        """Get current trust scores for specified nodes."""
        return {node: self.get_trust(node) for node in nodes}
    
    def get_reputation_metrics(self, node_id):
        """Get detailed reputation metrics for a node."""
        a = self.alpha.get(node_id, self.config['initial_alpha'])
        b = self.beta_param.get(node_id, self.config['initial_beta'])
        
        return {
            'trust_score': self.trust_scores.get(node_id, 0.5),
            'alpha': a,
            'beta': b,
            'variance': self._calculate_uncertainty(node_id),
            'total_evidence': a + b,
            'update_count': self.update_counts.get(node_id, 0)
        }
    
    def get_statistics(self):
        """Get system-wide trust statistics."""
        trusts = list(self.trust_scores.values())
        return {
            'mean_trust': np.mean(trusts),
            'std_trust': np.std(trusts),
            'min_trust': np.min(trusts),
            'max_trust': np.max(trusts),
            'median_trust': np.median(trusts),
            'total_updates': sum(self.update_counts.values()),
            'num_nodes': len(trusts)
        }
    
    def get_trust_distribution(self):
        """Get trust distribution for analysis."""
        return {
            'scores': list(self.trust_scores.values()),
            'alphas': list(self.alpha.values()),
            'betas': list(self.beta_param.values()),
            'node_ids': list(self.trust_scores.keys())
        }
    
    def reset(self):
        """Reset all trust scores to initial values."""
        for node_id in self.trust_scores:
            noise = random.uniform(-4, 4)
            self.alpha[node_id] = self.config['initial_alpha'] + noise
            self.beta_param[node_id] = self.config['initial_beta'] - noise
            total = self.alpha[node_id] + self.beta_param[node_id]
            self.trust_scores[node_id] = self.alpha[node_id] / total
            self.update_counts[node_id] = 0
            self.uncertainty[node_id] = self._calculate_uncertainty(node_id)
