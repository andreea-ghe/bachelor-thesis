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

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None) -> Tuple[Tensor, Tensor]:
        """
        Input:
            q: queries tensor [B, n_head, len_q, d_k]
            k: keys tensor [B, n_head, len_k, d_k]
            v: values tensor [B, n_head, len_v, d_v]
            mask: attention mask tensor [B, 1, len_k] or [B, len_q, len_k]
        
        Output:
            head_output: attended values [B, n_head, len_q, d_v]
            attn: attention weights [B, n_head, len_q, len_k]
        """
        # Compute attention scores: Q * K^T / sqrt(d_k)
        # we multiply by transpose of K because we want to compute dot product between each query and all keys
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

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

        self.n_head = n_head  # number of attention heads
        self.d_head = d_model // n_head  # dimension per head

        # Weights for queries, keys, values and output projection
        # Names must match checkpoint: w_qs, w_ks, w_vs, fc
        self.w_qs = nn.Linear(d_model, n_head * self.d_head, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * self.d_head, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * self.d_head, bias=False)
        self.fc = nn.Linear(n_head * self.d_head, d_model, bias=False)

        # Scaled dot-product attention
        self.attention = DotProductAttention(temperature=self.d_head ** 0.5, attn_dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None) -> Tuple[Tensor, Tensor]:
        """
        Input:
            q: queries tensor [B, len_q, d_model]
            k: keys tensor [B, len_k, d_model]
            v: values tensor [B, len_v, d_model]
            mask: attention mask tensor [B, len_q, len_k]
        
        Output:
            attended_features: attended values [B, len_q, d_model]
            attn: attention weights [B, n_head, len_q, len_k]
        """
        sz_b = q.size(0)
        len_q = q.size(1)
        len_k = k.size(1)
        len_v = v.size(1)

        residual = q  # we will add this back after attention; it's a residual connection

        q = self.w_qs(q).view(sz_b, len_q, self.n_head, self.d_head).transpose(1, 2)  # [B, n_head, len_q, d_head]
        k = self.w_ks(k).view(sz_b, len_k, self.n_head, self.d_head).transpose(1, 2)  # [B, n_head, len_k, d_head]
        v = self.w_vs(v).view(sz_b, len_v, self.n_head, self.d_head).transpose(1, 2)  # [B, n_head, len_v, d_head]

        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, len_q, len_k]
        
        # apply scaled dot-product attention
        head_output, attn = self.attention(q, k, v, mask=mask)  # head_output: [B, n_head, len_q, d_head]
        head_output = head_output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)  # [B, len_q, n_head * d_head]
        attended_features = self.fc(head_output)  # [B, len_q, d_model]
        
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
        # Names must match checkpoint: w_1, w_2
        self.w_1 = nn.Linear(d_input, d_hidden)
        self.w_2 = nn.Linear(d_hidden, d_input)
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

        output = self.dropout(self.w_2(F.relu(self.w_1(x))))

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
        # Names must match checkpoint: attn, pos_ffn
        self.attn = MultiHeadAttention(n_head=n_head, d_model=d_input, dropout=0.0)
        self.pos_ffn = PositionalFeedForwardNetwork(d_input=d_input, d_hidden=d_input * 2, dropout=0.0)

    def forward(self, x: Tensor) -> Tensor:
        """
        Input:
            x: input tensor [B, len_seq, d_input]
        
        Output:
            cross_features: transformed tensor [B, len_seq, d_input]
        """
        attn_output, attn_weights = self.attn(x, x, x)  # self-attention
        cross_features = self.pos_ffn(attn_output)

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
        # Name must match checkpoint: linear_p
        self.linear_p = nn.Sequential(
            nn.Linear(3, 3),
            LayerNorm1d(3),
            nn.ReLU(inplace=True),
            nn.Linear(3, out_features)
        )

        # MLP that produces the attention weight vector
        # Name must match checkpoint: linear_w
        self.linear_w = nn.Sequential(
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
        pos_enc = self.linear_p(relative_pos)  # [N, k, out_features]

        # Compute attention scores: (W_q - W_k + pos_enc)
        W_q_expanded = W_q.unsqueeze(1)  # [N, 1, mid_features]
        pos_enc = einops.reduce(pos_enc, "n ns (i j) -> n ns j", reduction="sum", j=self.mid_features)
        attn_scores = W_k - W_q_expanded + pos_enc  # [N, k, out_features]

        # compute attention weights
        weights = self.linear_w(attn_scores)  # [N, k, out_features // share_feat]
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
    # Tests both PointTransformer and CrossAttention
    
    pos = torch.randn(12, 3)
    x = torch.randn(12, 6)
    b = torch.tensor([4, 3, 5], dtype=torch.long).reshape(3, 1)

    pnf_layer = PointTransformer(
        in_features=6, out_features=6, n_heads=2, k_neighbors=4
    )
    x_pnf = pnf_layer(pos=pos, x=x, offset=b)
    print(x_pnf.shape)

    cross_attention_layer = CrossAttention(d_input=6, n_head=2)
    x = torch.randn(3, 4, 6)
    x_ca = cross_attention_layer(x)
    print(x_ca.shape)
