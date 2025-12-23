# # tdcb.py

# import torch
# from typing import Any, List, Dict

# # Auto-select device
# device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# class TDCB:
#     def __init__(self):
#         self.consensus_log: List[Dict] = []

#     def execute_consensus(self, transactions: List[Dict]) -> Dict[str, Any]:
#         """ Perform a trust-based delegated consensus mechanism on GPU if available. """
#         # print("Executing TDCB consensus...")

#         # Convert transactions to tensor
#         tx_count = torch.tensor(len(transactions), dtype=torch.float32, device=device)
#         verified_transactions = [tx for tx in transactions if self._verify_transaction(tx)]
#         verified_tx_count = torch.tensor(len(verified_transactions), dtype=torch.float32, device=device)

#         self.consensus_log.append(verified_transactions)

#         # print(f"Consensus complete. Verified transactions: {verified_tx_count.item()}")
#         return {"verified_transactions": verified_tx_count.item(), "status": "success"}

#     def _verify_transaction(self, transaction: Dict) -> bool:
#         """ Implement advanced verification logic. """
#         return transaction.get("amount", 0) > 0 and "sender" in transaction and "recipient" in transaction

import torch
from typing import Any, List, Dict, Set
import hashlib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TDCB:
    def __init__(self, fault_tolerance_ratio=0.33):
        self.consensus_log: List[Dict] = []
        self.fault_tolerance = fault_tolerance_ratio  # BFT: tolerate up to 33% malicious
        self.voting_history = []
        
    def execute_consensus(
        self, 
        transactions: List[Dict], 
        delegated_nodes: List[str],
        trust_scores: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Trust-weighted Byzantine Fault Tolerant consensus.
        
        Args:
            transactions: List of pending transactions
            delegated_nodes: Selected delegates for consensus
            trust_scores: Trust score for each delegate
        
        Returns:
            Consensus result with verified transactions
        """
        if not delegated_nodes:
            return {"verified_transactions": 0, "status": "no_delegates"}
        
        # Step 1: Each delegate votes on transactions
        votes = self._collect_votes(transactions, delegated_nodes, trust_scores)
        
        # Step 2: Aggregate votes with trust weighting
        verified_txs = self._aggregate_votes(votes, delegated_nodes, trust_scores, transactions)
        
        # Step 3: Byzantine fault detection
        malicious_delegates = self._detect_byzantine_behavior(votes, delegated_nodes)
        
        self.consensus_log.append({
            'verified_count': len(verified_txs),
            'total_submitted': len(transactions),
            'delegates': delegated_nodes,
            'malicious_detected': malicious_delegates
        })
        
        return {
            "verified_transactions": len(verified_txs),
            "status": "success",
            "consensus_round": len(self.consensus_log),
            "malicious_nodes": malicious_delegates,
            "verified_tx_list": verified_txs
        }
    
    def _collect_votes(
        self, 
        transactions: List[Dict], 
        delegates: List[str],
        trust_scores: Dict[str, float]
    ) -> Dict[str, Dict[str, bool]]:
        """Simulate each delegate voting on each transaction."""
        votes = {delegate: {} for delegate in delegates}
        
        for tx in transactions:
            tx_hash = self._hash_transaction(tx)
            for delegate in delegates:
                # Vote: delegate approves if transaction is valid
                # Simulate: higher trust = more likely to vote correctly
                trust = trust_scores.get(delegate, 0.5)
                is_valid = self._verify_transaction(tx)
                
                # Honest delegates vote correctly
                # Malicious delegates (low trust) may vote incorrectly
                if trust > 0.7:
                    votes[delegate][tx_hash] = is_valid
                elif trust < 0.3:
                    # Malicious: vote opposite
                    votes[delegate][tx_hash] = not is_valid
                else:
                    # Uncertain: random vote
                    votes[delegate][tx_hash] = (hash(delegate + tx_hash) % 2 == 0)
        
        return votes
    
    def _aggregate_votes(
        self,
        votes: Dict[str, Dict[str, bool]],
        delegates: List[str],
        trust_scores: Dict[str, float],
        transactions: List[Dict]  # ADD THIS PARAMETER
    ) -> List[Dict]:
        """
        Aggregate votes using trust-weighted majority.
        BFT requires 2/3+ majority with trust weighting.
        """
        verified_txs = []
        
        # Create hash-to-transaction mapping
        tx_hash_map = {self._hash_transaction(tx): tx for tx in transactions}
        
        # Group votes by transaction
        tx_votes = {}
        for delegate, tx_votes_dict in votes.items():
            for tx_hash, vote in tx_votes_dict.items():
                if tx_hash not in tx_votes:
                    tx_votes[tx_hash] = []
                tx_votes[tx_hash].append((delegate, vote))
        
        # Calculate weighted consensus
        for tx_hash, delegate_votes in tx_votes.items():
            total_weight = sum(trust_scores.get(d, 0.5) for d, _ in delegate_votes)
            approve_weight = sum(
                trust_scores.get(d, 0.5) for d, vote in delegate_votes if vote
            )
            
            # BFT threshold: 66.7% weighted approval
            if total_weight > 0 and approve_weight / total_weight >= 0.667:
                # Retrieve original transaction from hash map
                if tx_hash in tx_hash_map:
                    verified_txs.append(tx_hash_map[tx_hash])
        
        return verified_txs

    def _detect_byzantine_behavior(
        self,
        votes: Dict[str, Dict[str, bool]],
        delegates: List[str]
    ) -> List[str]:
        """
        Detect delegates voting significantly different from majority.
        Byzantine nodes consistently vote against consensus.
        """
        malicious = []
        
        # Calculate consensus vote for each transaction
        tx_hashes = set()
        for delegate_votes in votes.values():
            tx_hashes.update(delegate_votes.keys())
        
        for delegate in delegates:
            disagreement_count = 0
            total_votes = 0
            
            for tx_hash in tx_hashes:
                # Majority vote
                majority_vote = sum(
                    1 for d in delegates 
                    if votes.get(d, {}).get(tx_hash, False)
                ) > len(delegates) / 2
                
                # Delegate's vote
                delegate_vote = votes.get(delegate, {}).get(tx_hash, False)
                
                if delegate_vote != majority_vote:
                    disagreement_count += 1
                total_votes += 1
            
            # Flag as malicious if >50% disagreement
            if total_votes > 0 and disagreement_count / total_votes > 0.5:
                malicious.append(delegate)
        
        return malicious
    
    def _verify_transaction(self, transaction: Dict) -> bool:
        """Advanced transaction verification."""
        required_fields = ['amount', 'sender', 'recipient']
        if not all(field in transaction for field in required_fields):
            return False
        
        if transaction.get("amount", 0) <= 0:
            return False
        
        # Additional checks: signature verification, balance, etc.
        return True
    
    def _hash_transaction(self, tx: Dict) -> str:
        """Create unique hash for transaction."""
        tx_string = f"{tx.get('sender')}{tx.get('recipient')}{tx.get('amount')}"
        return hashlib.sha256(tx_string.encode()).hexdigest()[:16]
    
    def get_consensus_metrics(self):
        """Return consensus performance metrics."""
        if not self.consensus_log:
            return {}
        
        total_rounds = len(self.consensus_log)
        avg_verified = sum(log['verified_count'] for log in self.consensus_log) / total_rounds
        malicious_detected = sum(
            len(log['malicious_detected']) for log in self.consensus_log
        )
        
        return {
            'total_consensus_rounds': total_rounds,
            'avg_verified_per_round': avg_verified,
            'total_malicious_detected': malicious_detected,
            'consensus_success_rate': sum(
                1 for log in self.consensus_log if log['status'] == 'success'
            ) / total_rounds
        }
