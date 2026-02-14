import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple
import einops
from feature_extractor.utils import group_knn_features


class DotProductAttention(nn.Module):
    """
    Standard attention mechanism: Attention(Q, K, V) = softmax(QK^T / √d_k)V
    This is the core attention mechanism used in transformers.
    """
    def __init__(self, temperature, attn_dropout=0.0):
        super().__init__()
        self.temperature = temperature
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None, pair_bias: Tensor = None) -> Tuple[Tensor, Tensor]:
        """
        Input:
            q: queries tensor [B, n_head, len_q, d_k]
            k: keys tensor [B, n_head, len_k, d_k]
            v: values tensor [B, n_head, len_v, d_v]
            mask: attention mask tensor [B, 1, len_k] or [B, len_q, len_k]
            pair_bias: geometric pair attention bias [B, 1, len_q, len_k] (optional)
        
        Output:
            head_output: attended values [B, n_head, len_q, d_v]
            attn: attention weights [B, n_head, len_q, len_k]
        """
        # Compute attention scores: Q * K^T / sqrt(d_k)
        # we multiply by transpose of K because we want to compute dot product between each query and all keys
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        # Add pair attention bias: α_ij = n_ij + b_ij (Pair Attention)
        if pair_bias is not None:
            attn = attn + pair_bias

        # Apply mask if provided
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        # Apply softmax to get attention weights
        attn = F.softmax(attn, dim=-1)

        # Apply dropout to attention weights
        attn = self.attn_dropout(attn)

        # Compute attention weights * V
        head_output = torch.matmul(attn, v)

        return head_output, attn


class MultiHeadAttention(nn.Module):
    """
    Multiple attention heads allow the model to attend to different aspects of the input.
    """
    def __init__(self, n_head=8, d_model=512, dropout=0.0):
        super().__init__()

        self.n_head = n_head # number of attention heads
        self.d_head = d_model // n_head # dimension per head

        # Weights for queries, keys, values and output projection
        self.q_proj = nn.Linear(d_model, n_head * self.d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_head * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_head * self.d_head, bias=False)
        self.out_proj = nn.Linear(n_head * self.d_head, d_model, bias=False)

        # Scaled dot-product attention
        self.attention = DotProductAttention(temperature=self.d_head ** 0.5, attn_dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None, pair_bias: Tensor = None) -> Tuple[Tensor, Tensor]:
        """
        Input:
            q: queries tensor [B, len_q, d_model]
            k: keys tensor [B, len_k, d_model]
            v: values tensor [B, len_v, d_model]
            mask: attention mask tensor [B, len_q, len_k]
            pair_bias: geometric pair attention bias [B, 1, len_q, len_k] (optional)
        
        Output:
            attended_features: attended values [B, len_q, d_model]
            attn: attention weights [B, n_head, len_q, len_k]
        """
        sz_b = q.size(0)
        len_q = q.size(1)
        len_k = k.size(1)
        len_v = v.size(1)

        residual = q # we will add this back after attention; it's a residual connection

        q = self.q_proj(q).view(sz_b, len_q, self.n_head, self.d_head).transpose(1, 2)  # [B, n_head, len_q, d_head]
        k = self.k_proj(k).view(sz_b, len_k, self.n_head, self.d_head).transpose(1, 2)  # [B, n_head, len_k, d_head]
        v = self.v_proj(v).view(sz_b, len_v, self.n_head, self.d_head).transpose(1, 2)  # [B, n_head, len_v, d_head]

        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, len_q, len_k]
        
        # apply scaled dot-product attention
        head_output, attn = self.attention(q, k, v, mask=mask, pair_bias=pair_bias) # head_output: [B, n_head, len_q, d_head]
        head_output = head_output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)  # [B, len_q, n_head * d_head]
        attended_features = self.out_proj(head_output) # [B, len_q, d_model]
        
        attended_features = self.dropout(attended_features)
        attended_features += residual  # add residual connection
        attended_features = self.layer_norm(attended_features)  # normalize
        return attended_features, attn

class PositionalFeedForwardNetwork(nn.Module):
    """
    A two-layer feed-forward network with ReLU activation, applied independently
    to each position.
    FFN(x) = Linear(ReLU(Linear(x)))
    """
    def __init__(self, d_input, d_hidden, dropout=0.0):
        """
        Input:
            d_input: input feature dimension
            d_hidden: hidden layer dimension
            dropout: dropout rate
        """
        super().__init__()
        self.layer1 = nn.Linear(d_input, d_hidden)
        self.layer2 = nn.Linear(d_hidden, d_input)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_input, eps=1e-6)

    def forward(self, x: Tensor) -> Tensor:
        """
        Input:
            x: input tensor [B, len_seq, d_in]
        
        Output:
            output: transformed tensor [B, len_seq, d_in]
        """
        residual = x  # for residual connection

        output = self.dropout(self.layer2(F.relu(self.layer1(x))))
        # or output = self.dropout(self.layer2(self.dropout(F.relu(self.layer1(x))))))

        output += residual  # add residual connection
        output = self.layer_norm(output)  # normalize

        return output

class CrossAttention(nn.Module):
    """
    Combines multi-head attention with feed-forward network.
    Used for propagating information between pieces in the matching module.
    """
    def __init__(self, n_head, d_input):
        super(CrossAttention, self).__init__()
        self.multi_head_attn = MultiHeadAttention(n_head=n_head, d_model=d_input, dropout=0.0)
        self.ffn = PositionalFeedForwardNetwork(d_input=d_input, d_hidden=d_input * 2, dropout=0.0)

    def forward(self, x: Tensor, pair_bias: Tensor = None) -> Tensor:
        """
        Input:
            x: input tensor [B, len_seq, d_input]
            pair_bias: geometric pair attention bias [B, 1, len_seq, len_seq] (optional)
        
        Output:
            cross_features: transformed tensor [B, len_seq, d_input]
        """
        attn_output, attn_weights = self.multi_head_attn(x, x, x, pair_bias=pair_bias)  # self-attention + pair bias
        cross_features = self.ffn(attn_output)

        return cross_features

class LayerNorm1d(nn.BatchNorm1d):
    """
    1D Layer Normalization wrapper around BatchNorm1d
    
    Adapts BatchNorm1d to work with [N, C] shaped tensors by
    transposing to [N, C, 1] format expected by BatchNorm1d.
    
    Used in Point Transformer for normalizing point features.
    """
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()

class PointTransformer(nn.Module):
    """
    Implements a single Point Transformer layer that performs local feature aggregation
    using self-attention over k-nearest neighbors. This captures local geometric structure.
    The Point Transformer uses:
    1. K-NN to find local neighborhoods
    2. Position encoding to capture relative geometry
    3. Self-attention to aggregate neighbor features
    """

    def __init__(self, in_features, out_features, n_heads=8, k_neighbors=16):
        super(PointTransformer, self).__init__()
        self.in_features = in_features
        self.mid_features = out_features
        self.out_features = out_features
        self.share_feat = n_heads
        self.k_neighbors = k_neighbors

        # Linear layers to project input features to queries, keys, and values
        self.linear_q = nn.Linear(self.in_features, self.mid_features)
        self.linear_k = nn.Linear(self.in_features, self.mid_features)
        self.linear_v = nn.Linear(self.in_features, self.mid_features)

        # Positional encoding MLP: encodes relative 3D positions into features
        self.linear_pos = nn.Sequential(
            nn.Linear(3, 3),
            LayerNorm1d(3),
            nn.ReLU(inplace=True),
            nn.Linear(3, out_features)
        )

        # MLP_s is a mapping function that produces the weight vector
        self.MLP_s = nn.Sequential(
            LayerNorm1d(self.mid_features),
            nn.ReLU(inplace=True),
            nn.Linear(self.mid_features, self.out_features // self.share_feat),
            LayerNorm1d(self.out_features // self.share_feat),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_features // self.share_feat, self.out_features // self.share_feat)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, pos: Tensor, x: Tensor, offset: Tensor) -> Tensor:
        """
        Input:
            pos: point positions [N, 3]
            x: point features [N, in_features]
            offset: number of points per batch [B]
        """
        offset = offset.reshape(-1) # ensure offset is 1D tensor
        batch = torch.tensor(
            [b for b in range(len(offset)) for _ in range(offset[b])],
            dtype=torch.long,
            device=x.device
        )

        # Project to Q, K, V
        W_q = self.linear_q(x)
        W_k = self.linear_k(x)
        W_v = self.linear_v(x)

        W_k, _ = group_knn_features(
            all_features=W_k, 
            all_coords=pos, 
            k=self.k_neighbors, 
            query_coords=None, 
            batch_idx=batch, 
            include_relative_pos=True
        ) # [N, k, mid_features + 3]
        W_v, _ = group_knn_features(
            all_features=W_v, 
            all_coords=pos, 
            k=self.k_neighbors, 
            query_coords=None, 
            batch_idx=batch, 
            include_relative_pos=False
        ) # [N, k, mid_features]

        # Separate relative positions from W_k features
        relative_pos = W_k[:, :, :3]  # [N, k, 3] 
        W_k = W_k[:, :, 3:]  # [N, k, mid_features]

        # Compute positional encodings
        pos_enc = self.linear_pos(relative_pos)  # [N, k, out_features]

        # Compute attention scores: (W_q - W_k + pos_enc)
        W_q_expanded = W_q.unsqueeze(1)  # [N, 1, mid_features]
        pos_enc = einops.reduce(pos_enc, "n ns (i j) -> n ns j", reduction="sum", j=self.mid_features)
        attn_scores =  W_k - W_q_expanded + pos_enc  # [N, k, out_features]

        # compute attention weights
        weights = self.MLP_s(attn_scores)  # [N, k, out_features // share_feat]
        weights = self.softmax(weights)  # [N, k, out_features // share_feat]

        # Aggregate features
        x = torch.einsum(
            "n t s i, n t i -> n s i",
            einops.rearrange(
                W_v + pos_enc,
                "n ns (s i) -> n ns s i",
                s=self.share_feat
            ),
            weights
        )
        x = einops.rearrange(
            x,
            "n s i -> n (s i)"
        )  # [N, out_features]

        return x

if __name__ == "__main__":
    # Minimum test example to verify the attention layers work correctly
    # Tests PointTransformer, CrossAttention without bias, and CrossAttention with pair bias
    
    pos = torch.randn(12, 3)
    x = torch.randn(12, 6)
    b = torch.tensor([4, 3, 5], dtype=torch.long).reshape(3, 1)

    pnf_layer = PointTransformer(
        in_features=6, out_features=6, n_heads=2, k_neighbors=4
    )
    x_pnf = pnf_layer(pos=pos, x=x, offset=b)
    print(f"PointTransformer: {x_pnf.shape}")

    cross_attention_layer = CrossAttention(d_input=6, n_head=2)
    x = torch.randn(3, 4, 6)

    # Without pair bias (original behavior)
    x_ca = cross_attention_layer(x)
    print(f"CrossAttention (no bias): {x_ca.shape}")

    # With pair bias (pair attention)
    pair_bias = torch.randn(3, 1, 4, 4)  # [B, 1, N_SUM, N_SUM]
    x_ca_biased = cross_attention_layer(x, pair_bias=pair_bias)
    print(f"CrossAttention (with pair bias): {x_ca_biased.shape}")
