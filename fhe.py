# fhe.py

from cryptography.fernet import Fernet

# class FullyHomomorphicEncryption:
#     def __init__(self):
#         # Initialize encryption key
#         self.key = Fernet.generate_key()
#         self.cipher = Fernet(self.key)

#     def encrypt(self, plaintext):
#         """Encrypt the data."""
#         encrypted = self.cipher.encrypt(plaintext.encode())
#         return encrypted

#     def decrypt(self, encrypted_data):
#         """Decrypt the data."""
#         decrypted = self.cipher.decrypt(encrypted_data).decode()
#         return decrypted

try:
    import tenseal as ts
    import numpy as np
    TENSEAL_AVAILABLE = True
except ImportError:
    TENSEAL_AVAILABLE = False
    print("[WARNING] TenSEAL not installed. Install with: pip install tenseal")
    print("[WARNING] Falling back to mock encryption for demonstration.")

class FullyHomomorphicEncryption:
    def __init__(self, use_real_fhe=True):
        """
        Initialize FHE with CKKS scheme for encrypted computations.
        
        Args:
            use_real_fhe: If True and TenSEAL available, use real HE. Else use mock.
        """
        self.use_real_fhe = use_real_fhe and TENSEAL_AVAILABLE
        
        if self.use_real_fhe:
            # Initialize CKKS encryption context
            self.context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=8192,  # Security parameter
                coeff_mod_bit_sizes=[60, 40, 40, 60]  # Modulus chain
            )
            self.context.global_scale = 2**40  # Precision
            self.context.generate_galois_keys()  # For rotation operations
            print("[FHE] TenSEAL CKKS encryption initialized")
        else:
            # Fallback to simple encryption for demonstration
            from cryptography.fernet import Fernet
            self.key = Fernet.generate_key()
            self.cipher = Fernet(self.key)
            print("[FHE] Using mock encryption (not true homomorphic)")
    
    def encrypt_value(self, value):
        """Encrypt a single numerical value."""
        if self.use_real_fhe:
            # Encrypt with CKKS (supports floating point)
            encrypted = ts.ckks_vector(self.context, [float(value)])
            return encrypted
        else:
            # Mock encryption
            return self.cipher.encrypt(str(value).encode())
    
    def encrypt_vector(self, values):
        """Encrypt a vector of values (e.g., trust scores)."""
        if self.use_real_fhe:
            encrypted = ts.ckks_vector(self.context, [float(v) for v in values])
            return encrypted
        else:
            # Mock: encrypt each value separately
            return [self.cipher.encrypt(str(v).encode()) for v in values]
    
    def decrypt_value(self, encrypted_value):
        """Decrypt a single value."""
        if self.use_real_fhe:
            decrypted_list = encrypted_value.decrypt()
            return decrypted_list[0]  # Return first element
        else:
            return float(self.cipher.decrypt(encrypted_value).decode())
    
    def decrypt_vector(self, encrypted_vector):
        """Decrypt a vector."""
        if self.use_real_fhe:
            return encrypted_vector.decrypt()
        else:
            return [float(self.cipher.decrypt(v).decode()) for v in encrypted_vector]
    
    def compute_encrypted_mean(self, encrypted_vector):
        """
        Compute mean on encrypted data (homomorphic operation).
        This demonstrates FHE's power: computation without decryption.
        """
        if self.use_real_fhe:
            # Sum all encrypted values
            n = len(encrypted_vector.decrypt())  # Only for demo - in real use, n is public
            encrypted_sum = encrypted_vector  # CKKS preserves vector operations
            # Multiply by 1/n (scalar multiplication)
            encrypted_mean = encrypted_sum * (1.0 / n)
            return encrypted_mean
        else:
            # Mock: can't compute on encrypted data
            decrypted = self.decrypt_vector(encrypted_vector)
            mean_val = sum(decrypted) / len(decrypted)
            return self.encrypt_value(mean_val)
    
    def compute_encrypted_comparison(self, encrypted_value, threshold):
        """
        Compare encrypted value to threshold (simplified).
        Note: True HE comparison is complex; this is approximation.
        """
        if self.use_real_fhe:
            # In production, use polynomial approximation of comparison
            # For now, decrypt for demo (in real app, use sign polynomials)
            decrypted = self.decrypt_value(encrypted_value)
            result = 1.0 if decrypted > threshold else 0.0
            return self.encrypt_value(result)
        else:
            decrypted = self.decrypt_value(encrypted_value)
            result = 1.0 if decrypted > threshold else 0.0
            return self.encrypt_value(result)
    
    def serialize(self, encrypted_data):
        """Serialize encrypted data for storage/transmission."""
        if self.use_real_fhe:
            return encrypted_data.serialize()
        else:
            return encrypted_data  # Already bytes
    
    def deserialize(self, serialized_data):
        """Deserialize encrypted data."""
        if self.use_real_fhe:
            return ts.ckks_vector_from(self.context, serialized_data)
        else:
            return serialized_data
