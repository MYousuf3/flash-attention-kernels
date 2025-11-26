import torch
import math
import torch.nn.functional as F

class CustomSelfAttention(torch.nn.Module):
    '''
    Custom self-attention implementation.

    Args:
        w_q: Query weight matrix of shape (hidden_dim, hidden_dim)
        w_k: Key weight matrix of shape (hidden_dim, hidden_dim)
        w_v: Value weight matrix of shape (hidden_dim, hidden_dim)
        w_o: Output weight matrix of shape (hidden_dim, hidden_dim)
        num_heads: Number of attention heads
    '''
    def __init__(self, w_q, w_k, w_v, w_o, hidden_dim, num_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.w_q = w_q
        self.w_k = w_k
        self.w_v = w_v
        self.w_o = w_o

    def forward(self, x, causal=False):
        '''
        Forward pass for the self-attention layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            causal: Whether to use causal attention
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_dim)
        '''
    
        batch_size, seq_len, _ = x.shape

        # Obtain the query, key, and value tensors
        q = x @ self.w_q.T
        k = x @ self.w_k.T
        v = x @ self.w_v.T

        ##############################################################

        # TODO: Implement the self-attention operation
        # transforming the heads
        q_prime = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k_prime = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v_prime = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        q_prime = q_prime.transpose(1, 2)  
        k_prime = k_prime.transpose(1, 2)  
        v_prime = v_prime.transpose(1, 2)  


        qk_t = q_prime @ k_prime.transpose(-2, -1)

        root_d = math.sqrt(self.head_dim)

        score = qk_t / root_d

        if causal:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            score = score.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(score, dim=-1)
        
        o = attn_weights @ v_prime

        ##############################################################

        # Concatenate attention heads
        o = o.transpose(1, 2)
        o = o.contiguous().view(batch_size, seq_len, self.hidden_dim)

        # Apply output projection
        o = o @ self.w_o.T
        return o