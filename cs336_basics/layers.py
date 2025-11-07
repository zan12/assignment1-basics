from einops import einsum, rearrange
from jaxtyping import Float, Int
from torch import Tensor

import math
import torch
import torch.nn as nn


class Linear(nn.Module):

    def __init__(
            self,
            d_in: int,
            d_out: int,
            device: torch.device = None,
            dtype: torch.dtype = None,
        ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.device = device
        self.dtype = dtype
        # self.weight = nn.Parameter(torch.empty(d_out, d_in)).to(device=self.device, dtype=self.dtype)
        self.truncated_normal_init()
 
    def truncated_normal_init(self,):
        std = math.sqrt(2/(self.d_in + self.d_out))
        init_weight = nn.init.trunc_normal_(
            tensor = torch.empty(self.d_out, self.d_in),
            std = std,
            a = -3 * std,
            b = 3 * std,
        )
        self.weight = nn.Parameter(init_weight).to(device=self.device, dtype=self.dtype)
    
    def forward(self, x: torch.Tensor):
        return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")
    

class Embedding(nn.Module):
    
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        # self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim)).to(device=device, dtype=dtype)
        self.truncated_normal_init()

    def truncated_normal_init(self,):
        init_weight = nn.init.trunc_normal_(
            tensor = torch.empty(self.num_embeddings, self.embedding_dim),
            a = -3,
            b = 3,
        )
        self.weight = nn.Parameter(init_weight).to(device=self.device, dtype=self.dtype)
    
    def forward(self, token_ids: torch.Tensor):
        return self.weight[token_ids,...]
    

class RMSNorm(nn.Module):
    
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        # self.weight = nn.Parameter(torch.empty(d_model)).to(device=device, dtype=dtype)
        self.constant_init()

    def constant_init(self,):
        weight = torch.ones(self.d_model)
        self.weight = nn.Parameter(weight.to(device=self.device, dtype=self.dtype))

    def forward(self, x: torch.Tensor):
        x_dtype = x.dtype
        x = x.to(dtype=torch.float32)
        normalizer = torch.sqrt(einsum(x.pow(2), "... d_model -> ...") / self.d_model + self.eps)
        x = x / rearrange(normalizer, "... -> ... 1")
        result = self.weight * x
        return result.to(dtype=x_dtype)
    

class SiLU(nn.Module):

    def __init__(self,):
        super().__init__()

    def forward(self, in_features):
        return in_features * torch.sigmoid(in_features)


class SwiGLU(nn.Module):
    
    def __init__(self, d_model: int, d_ff: int, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype
        self.silu = SiLU()
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)
        
    def forward(self, x: torch.Tensor):
        return self.w2(self.silu(self.w1(x)) * self.w3(x))
    

class RoPE(nn.Module):
    
    def __init__(self, d_k: int, theta: float, max_seq_len: int, device: torch.device = None):
        super().__init__()
        assert d_k % 2 == 0
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.device = device
        unit_angle = torch.Tensor([1/pow(theta, 2*k/d_k) for k in range(d_k//2)])
        angles = torch.outer(torch.arange(max_seq_len), unit_angle) # [max_seq_len, d_k//2]
        self.cos_theta = torch.cos(angles)
        self.sin_theta = torch.sin(angles)
        # self.register_buffer("cos_theta", torch.cos(angles))
        # self.register_buffer("sin_theta", torch.sin(angles))

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape
        x = x.view(*x.shape[:-1], x.shape[-1]//2, 2)
        cos_theta = self.cos_theta[token_positions,:]
        sin_theta = self.sin_theta[token_positions,:]
        x_w_rope = torch.stack([cos_theta * x[..., 0], cos_theta * x[..., 1]], dim=-1) + torch.stack([-sin_theta * x[..., 1], sin_theta * x[..., 0]], dim=-1)
        x_w_rope = x_w_rope.view(x_shape)
        return x_w_rope

    
def softmax(x: torch.Tensor, dim: int = -1):
    x_max, _ = torch.max(x, dim = dim, keepdim=True)
    exp_x = torch.exp(x - x_max)
    return exp_x / torch.sum(exp_x, dim=-1, keepdim=True)


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor):
    d_k = q.shape[-1]
    assert k.shape[-1] == d_k
    qk_d = einsum(q, k, "... t d_k, ... s d_k -> ... t s") / math.sqrt(d_k)
    additive_mask = torch.where(mask, 0.0, float('-inf'))
    qk_d += additive_mask
    return einsum(softmax(qk_d), v, "... t s, ... s d_v -> ... t d_v")


class MultiHeadSelfAttention(nn.Module):
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float = None,
        max_seq_len: int = None,
        use_rope: bool = True,
        merge_qkv_proj = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.merge_qkv_proj = merge_qkv_proj
        if merge_qkv_proj:
            self.qkv_proj = Linear(d_in=d_model, d_out=3*d_model)
        else:
            self.q_proj = Linear(d_in=d_model, d_out=d_model)
            self.k_proj = Linear(d_in=d_model, d_out=d_model)
            self.v_proj = Linear(d_in=d_model, d_out=d_model)
        self.output_proj = Linear(d_in=d_model, d_out=d_model)
        self.use_rope = use_rope
        self.rope = RoPE(d_model//num_heads, theta, max_seq_len) if use_rope else None


    def forward(self, in_features, token_positions=None):
        if self.merge_qkv_proj:
            qkv_proj = self.qkv_proj(in_features) # b t 3d
        else:
            qkv_proj = torch.cat(
                [
                    self.q_proj(in_features), 
                    self.k_proj(in_features), 
                    self.v_proj(in_features),
                ],
                dim=-1,
            ) # b t 3d
        qkv_proj = qkv_proj.view(*qkv_proj.shape[:-1], 3 * self.num_heads, -1) # b t 3n h
        qkv_proj = rearrange(qkv_proj, "... t n h -> ... n t h") # b 3n t h
        q_proj, k_proj, v_proj = torch.chunk(qkv_proj, chunks=3, dim=-3) # b n t h
        t = in_features.shape[-2]
        if self.use_rope: # Apple RoPE to k and v.
            if token_positions == None:
                token_positions = torch.range(0,t-1).repeat(*in_features.shape[:-2],1).to(dtype=torch.int32)
            token_positions = torch.stack([token_positions]*self.num_heads, dim=-2) # b t -> b n t
            q_proj = self.rope(q_proj, token_positions)
            k_proj = self.rope(k_proj, token_positions)
        # Masked sdpa.
        mask = (torch.tril(torch.ones(t,t)) == 1).to(in_features.device)
        qkv = scaled_dot_product_attention(q_proj,k_proj,v_proj,mask) # btnh
        qkv = rearrange(qkv, "b n t h -> b t (n h)")
        return self.output_proj(qkv)
    

class TransformerBlock(nn.Module):
    
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, use_rope, theta):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, theta, max_seq_len, use_rope)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)
    
    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
    

class TransformerLM(nn.Module):
    
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        use_rope: bool,
        rope_theta: float,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model,
                num_heads,
                d_ff,
                context_length,
                use_rope,
                rope_theta,
            ) for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, in_indices: torch.Tensor):
        x = self.token_embeddings(in_indices)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.lm_head(x)
    

def cross_entropy(
    logits: Float[Tensor, " ... vocab_size"],
    targets: Int[Tensor, " ..."]
):
    vocab_size = logits.shape[-1]
    logits = logits.view(-1, vocab_size)
    targets = targets.view(-1)
    logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
    logits -= logits_max
    sum_exp_logits = torch.sum(torch.exp(logits), dim = -1)
    return -torch.mean(logits[torch.arange(logits.size(0)),targets] - torch.log(sum_exp_logits))
