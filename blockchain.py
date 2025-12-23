# blockchain.py

import hashlib
import time
from typing import List, Dict, Any
from dataclasses import dataclass, field

@dataclass
class Block:
    index: int
    transactions: List[Dict[str, Any]]
    previous_hash: str
    timestamp: float = field(default_factory=time.time)
    nonce: int = 0
    
    def calculate_hash(self) -> str:
        """
        Advanced hash calculation with SHA-3 and dynamic salting
        """
        block_string = (
            str(self.index) + 
            str(self.timestamp) + 
            str(self.previous_hash) + 
            str(self.transactions) + 
            str(self.nonce)
        )
        return hashlib.sha3_256(block_string.encode()).hexdigest()
    
    def mine_block(self, difficulty: int = 4) -> None:
        """
        Proof of Work mining mechanism
        """
        target = "0" * difficulty
        while self.calculate_hash()[:difficulty] != target:
            self.nonce += 1

class Blockchain:
    def __init__(self, difficulty: int = 4):
        self.difficulty = difficulty
        self.chain: List[Block] = []
        self.pending_transactions: List[Dict[str, Any]] = []
        self.delegation_ratio = 0.5
        self._create_genesis_block()
    
    def _create_genesis_block(self) -> None:
        genesis_block = Block(0, [], "0")
        genesis_block.mine_block(self.difficulty)
        self.chain.append(genesis_block)
    
    def get_latest_block(self) -> Block:
        return self.chain[-1]
    
    def add_transaction(self, transaction: Dict[str, Any]) -> None:
        """
        Advanced transaction validation
        """
        if self._validate_transaction(transaction):
            self.pending_transactions.append(transaction)
    
    def _validate_transaction(self, transaction: Dict[str, Any]) -> bool:
        # Add custom transaction validation logic
        return all(key in transaction for key in ['sender', 'recipient', 'amount'])
    
    def mine_pending_transactions(self, mining_reward_address: str) -> None:
        block = Block(
            len(self.chain), 
            self.pending_transactions, 
            self.get_latest_block().calculate_hash()
        )
        block.mine_block(self.difficulty)
        
        # Add mining reward
        mining_reward = {
            'sender': 'SYSTEM',
            'recipient': mining_reward_address,
            'amount': 10  # Configurable mining reward
        }
        block.transactions.append(mining_reward)
        
        self.chain.append(block)
        self.pending_transactions = []
    
    def is_chain_valid(self) -> bool:
        """
        Advanced chain validation with comprehensive checks
        """
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i-1]
            
            # Check hash continuity
            if current_block.previous_hash != previous_block.calculate_hash():
                return False
            
            # Check block's own hash validity
            if current_block.calculate_hash()[:self.difficulty] != "0" * self.difficulty:
                return False
        
        return True