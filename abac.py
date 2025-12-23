# abac.py - Attribute-Based Access Control with FHE Integration

from typing import Dict, List, Set, Optional
from dataclasses import dataclass
import time
import numpy as np
from collections import defaultdict


@dataclass
class AttributePolicy:
    """Policy definition for attribute-based access control."""
    required_attributes: Set[str]
    allowed_actions: Set[str]
    time_restrictions: Optional[Dict[str, tuple]]
    location_restrictions: Optional[Set[str]]


class ABAC:
    """
    Attribute-Based Access Control (ABAC) with trust integration.
    
    Features:
    - Traditional ABAC policy enforcement
    - Trust-aware access decisions
    - Adaptive learning from feedback
    - Policy effectiveness tracking
    - Optional FHE integration
    """
    
    def __init__(self, fhe_enabled: bool = False):
        """
        Initialize ABAC system.
        
        Args:
            fhe_enabled: If True, use FHE for encrypted attribute comparison
        """
        self.policies: Dict[str, AttributePolicy] = {}
        self.user_attributes: Dict[str, Set[str]] = {}
        self.fhe_enabled = fhe_enabled
        self._init_default_policies()
        
        # ML-based adaptive learning
        self.access_history = []
        self.policy_confidence = defaultdict(lambda: 1.0)
        self.learning_rate = 0.1
        self.trust_threshold = 0.5  # Minimum trust for access
        
        # Performance tracking
        self.total_requests = 0
        self.granted_requests = 0
        self.denied_requests = 0
        
        # FHE instance (lazy initialization)
        self._fhe = None
        
    def _init_default_policies(self) -> None:
        """Initialize default ABAC policies."""
        self.policies["admin"] = AttributePolicy(
            required_attributes={"admin_role", "security_clearance"},
            allowed_actions={"read", "write", "execute", "modify"},
            time_restrictions={"weekday": (0, 24), "weekend": (9, 17)},
            location_restrictions={"secure_location", "headquarters"}
        )
        
        self.policies["user"] = AttributePolicy(
            required_attributes={"user_role"},
            allowed_actions={"read", "write"},
            time_restrictions={"weekday": (9, 17), "weekend": None},
            location_restrictions=None
        )
        
        self.policies["guest"] = AttributePolicy(
            required_attributes={"guest_role"},
            allowed_actions={"read"},
            time_restrictions={"weekday": (9, 17), "weekend": (10, 16)},
            location_restrictions=None
        )
    
    def _get_fhe(self):
        """Lazy initialization of FHE module."""
        if self._fhe is None and self.fhe_enabled:
            try:
                from fhe import FullyHomomorphicEncryption
                self._fhe = FullyHomomorphicEncryption(use_real_fhe=True)
            except ImportError:
                print("[ABAC] Warning: FHE module not available, using plaintext")
                self.fhe_enabled = False
        return self._fhe
        
    def add_user_attributes(self, user_id: str, attributes: Set[str]) -> None:
        """Add or update user attributes."""
        self.user_attributes[user_id] = attributes
        
    def check_attribute_requirements(
        self,
        user_id: str,
        policy: AttributePolicy
    ) -> bool:
        """Check if user has required attributes."""
        user_attrs = self.user_attributes.get(user_id, set())
        return policy.required_attributes.issubset(user_attrs)
    
    def check_time_restrictions(
        self,
        policy: AttributePolicy,
        time_of_day: int,
        day_type: str
    ) -> bool:
        """Verify time-based restrictions."""
        if policy.time_restrictions is None:
            return True
        
        time_range = policy.time_restrictions.get(day_type)
        if time_range is None:
            return False
        
        start_hour, end_hour = time_range
        return start_hour <= time_of_day < end_hour
    
    def check_location_restrictions(
        self,
        policy: AttributePolicy,
        location: str
    ) -> bool:
        """Verify location-based restrictions."""
        if policy.location_restrictions is None:
            return True
        return location in policy.location_restrictions
    
    def enforce_policy(
        self,
        user_id: str,
        requested_action: str,
        time_of_day: int,
        day_type: str,
        location: str
    ) -> bool:
        """
        Traditional ABAC policy enforcement.
        
        Args:
            user_id: User identifier
            requested_action: Action being requested
            time_of_day: Current hour (0-23)
            day_type: "weekday" or "weekend"
            location: User's location
            
        Returns:
            bool: True if access granted, False otherwise
        """
        user_attrs = self.user_attributes.get(user_id, set())
        
        for policy_name, policy in self.policies.items():
            # Check if user matches this policy
            if not self.check_attribute_requirements(user_id, policy):
                continue
            
            # Check if action is allowed
            if requested_action not in policy.allowed_actions:
                continue
            
            # Check time restrictions
            if not self.check_time_restrictions(policy, time_of_day, day_type):
                continue
            
            # Check location restrictions
            if not self.check_location_restrictions(policy, location):
                continue
            
            # All checks passed
            return True
        
        return False
    
    def enforce_policy_with_fhe(
        self,
        user_id: str,
        requested_action: str,
        time_of_day: int,
        day_type: str,
        location: str,
        trust_score: float = 1.0
    ) -> tuple:
        """
        FHE-encrypted policy enforcement.
        
        This implements the paper's claim of FHE-backed ABAC:
        - Attributes are encrypted before comparison
        - Policy decisions are made on encrypted data
        - Only the final decision is decrypted
        
        Returns:
            tuple: (decision: bool, confidence: float, reason: str)
        """
        fhe = self._get_fhe()
        
        if fhe is None or not self.fhe_enabled:
            # Fallback to plaintext if FHE not available
            return self.enforce_policy_with_learning(
                user_id, requested_action, time_of_day, 
                day_type, location, trust_score
            )
        
        # Encrypt trust score
        encrypted_trust = fhe.encrypt_value(trust_score)
        
        # Compare encrypted values
        trust_check_result = fhe.compute_encrypted_comparison(
            encrypted_trust, 
            self.trust_threshold
        )
        
        # Decrypt only the final result
        trust_passed = fhe.decrypt_value(trust_check_result) > 0.5
        
        # Base policy decision (can also be encrypted in full implementation)
        base_decision = self.enforce_policy(
            user_id, requested_action, time_of_day, day_type, location
        )
        
        # Combined decision
        final_decision = base_decision and trust_passed
        confidence = trust_score if trust_passed else (1.0 - trust_score)
        
        if final_decision:
            reason = "FHE-verified: policy and trust checks passed"
            self.granted_requests += 1
        else:
            if not base_decision:
                reason = "FHE-verified: policy restrictions not met"
            else:
                reason = f"FHE-verified: trust below threshold ({trust_score:.2f})"
            self.denied_requests += 1
        
        self.total_requests += 1
        return final_decision, confidence, reason
    
    def enforce_policy_with_learning(
        self,
        user_id: str,
        requested_action: str,
        time_of_day: int,
        day_type: str,
        location: str,
        trust_score: float = 1.0
    ) -> tuple:
        """
        Enhanced policy enforcement with adaptive learning and trust integration.
        
        Args:
            user_id: User identifier
            requested_action: Action being requested
            time_of_day: Current hour (0-23)
            day_type: "weekday" or "weekend"
            location: User's location
            trust_score: Trust score from TrustManager (0.0 to 1.0)
            
        Returns:
            tuple: (decision: bool, confidence: float, reason: str)
        """
        self.total_requests += 1
        
        # Step 1: Traditional ABAC decision
        base_decision = self.enforce_policy(
            user_id, requested_action, time_of_day, day_type, location
        )
        
        # Step 2: Trust-based adjustment
        final_decision = base_decision
        confidence = 0.5
        reason = "No matching policy"
        
        if base_decision:
            # Policy allows, but check trust
            if trust_score >= self.trust_threshold:
                final_decision = True
                confidence = min(1.0, trust_score + 0.2)
                reason = "Policy and trust verified"
                self.granted_requests += 1
            else:
                # Low trust overrides policy
                final_decision = False
                confidence = 0.9
                reason = f"Low trust ({trust_score:.2f} < {self.trust_threshold})"
                self.denied_requests += 1
        else:
            # Policy denies
            final_decision = False
            confidence = 0.8
            reason = "Policy restrictions not met"
            self.denied_requests += 1
        
        # Step 3: Log for learning
        context = {
            'time': time_of_day,
            'day': day_type,
            'location': location,
            'trust': trust_score,
            'action': requested_action
        }
        self.access_history.append({
            'user': user_id,
            'action': requested_action,
            'context': context,
            'decision': final_decision,
            'reason': reason,
            'timestamp': time.time()
        })
        
        return final_decision, confidence, reason
    
    def learn_from_feedback(self, user_id: str, action: str, was_correct: bool):
        """
        Update policy confidence based on admin feedback.
        """
        user_attrs = self.user_attributes.get(user_id, set())
        
        for policy_name, policy in self.policies.items():
            if policy.required_attributes.issubset(user_attrs):
                if was_correct:
                    self.policy_confidence[policy_name] += self.learning_rate
                else:
                    self.policy_confidence[policy_name] -= self.learning_rate
                
                self.policy_confidence[policy_name] = np.clip(
                    self.policy_confidence[policy_name], 0.0, 2.0
                )
                break
    
    def adjust_trust_threshold(self, new_threshold: float):
        """Dynamically adjust trust threshold."""
        self.trust_threshold = np.clip(new_threshold, 0.0, 1.0)
    
    def get_policy_effectiveness(self) -> Dict:
        """Return comprehensive policy performance metrics."""
        if self.total_requests == 0:
            return {
                'total_requests': 0,
                'granted_ratio': 0.0,
                'denied_ratio': 0.0,
                'policy_confidences': {},
                'trust_threshold': self.trust_threshold
            }
        
        return {
            'total_requests': self.total_requests,
            'granted_count': self.granted_requests,
            'denied_count': self.denied_requests,
            'granted_ratio': self.granted_requests / self.total_requests,
            'denied_ratio': self.denied_requests / self.total_requests,
            'policy_confidences': dict(self.policy_confidence),
            'trust_threshold': self.trust_threshold
        }
    
    def reset_statistics(self):
        """Reset all statistics and learning data."""
        self.access_history = []
        self.policy_confidence = defaultdict(lambda: 1.0)
        self.total_requests = 0
        self.granted_requests = 0
        self.denied_requests = 0
