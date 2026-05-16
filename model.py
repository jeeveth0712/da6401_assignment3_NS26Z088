"""
model.py — Transformer Architecture Skeleton
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────┐
  │  scaled_dot_product_attention(Q, K, V, mask) → (out, weights)  │
  │  MultiHeadAttention.forward(q, k, v, mask)   → Tensor          │
  │  PositionalEncoding.forward(x)               → Tensor          │
  │  make_src_mask(src, pad_idx)                 → BoolTensor      │
  │  make_tgt_mask(tgt, pad_idx)                 → BoolTensor      │
  │  Transformer.encode(src, src_mask)           → Tensor          │
  │  Transformer.decode(memory,src_m,tgt,tgt_m)  → Tensor          │
  └─────────────────────────────────────────────────────────────────┘
"""

import math
import copy
import os
import re
import gdown
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Fill in your Google Drive file ID after uploading your trained checkpoint ──
GDRIVE_FILE_ID = "1aB13TQT5epPet7AkCDc_HD4d-vk9_vtB"


def _tok_de(text: str):
    return re.findall(r"[a-zA-ZäöüÄÖÜß\-]+|[^\w\s]", text.lower())


def _tok_en(text: str):
    return re.findall(r"[a-zA-Z\-]+|[^\w\s]", text.lower())


# ══════════════════════════════════════════════════════════════════════
#   STANDALONE ATTENTION FUNCTION
# ══════════════════════════════════════════════════════════════════════


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V

    Args:
        Q    : (..., seq_q, d_k)
        K    : (..., seq_k, d_k)
        V    : (..., seq_k, d_v)
        mask : BoolTensor broadcastable to (..., seq_q, seq_k).
               True → masked out (set to -inf before softmax).
    Returns:
        output : (..., seq_q, d_v)
        attn_w : (..., seq_q, seq_k)
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    attn_w = F.softmax(scores, dim=-1)
    # replace NaN (all-masked rows) with 0
    attn_w = torch.nan_to_num(attn_w, nan=0.0)
    output = torch.matmul(attn_w, V)
    return output, attn_w


# ══════════════════════════════════════════════════════════════════════
#   MASK HELPERS
# ══════════════════════════════════════════════════════════════════════


def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Padding mask for the encoder.

    Returns:
        BoolTensor [batch, 1, 1, src_len]  — True where PAD.
    """
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Combined padding + causal (look-ahead) mask for the decoder.

    Returns:
        BoolTensor [batch, 1, tgt_len, tgt_len]  — True where masked out.
    """
    tgt_len = tgt.size(1)
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)  # [B,1,1,T]
    causal = (
        torch.triu(
            torch.ones(tgt_len, tgt_len, device=tgt.device, dtype=torch.bool),
            diagonal=1,
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )  # [1,1,T,T]
    return pad_mask | causal  # [B,1,T,T]


# ══════════════════════════════════════════════════════════════════════
#   MULTI-HEAD ATTENTION
# ══════════════════════════════════════════════════════════════════════


class MultiHeadAttention(nn.Module):
    """
    MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W_O
    head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query : [batch, seq_q, d_model]
            key   : [batch, seq_k, d_model]
            value : [batch, seq_k, d_model]
            mask  : BoolTensor broadcastable to [batch, num_heads, seq_q, seq_k]
        Returns:
            output : [batch, seq_q, d_model]
        """
        B = query.size(0)

        def project(linear, x):
            return linear(x).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)

        Q = project(self.W_q, query)  # [B, H, seq_q, d_k]
        K = project(self.W_k, key)  # [B, H, seq_k, d_k]
        V = project(self.W_v, value)  # [B, H, seq_k, d_k]

        out, _ = scaled_dot_product_attention(Q, K, V, mask)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.W_o(out)


# ══════════════════════════════════════════════════════════════════════
#   POSITIONAL ENCODING
# ══════════════════════════════════════════════════════════════════════


class PositionalEncoding(nn.Module):
    """
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [batch, seq_len, d_model]
        Returns:
            [batch, seq_len, d_model]
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
#   FEED-FORWARD NETWORK
# ══════════════════════════════════════════════════════════════════════


class PositionwiseFeedForward(nn.Module):
    """FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════════
#   ENCODER LAYER
# ══════════════════════════════════════════════════════════════════════


class EncoderLayer(nn.Module):
    """x → [Self-Attn → Add & Norm] → [FFN → Add & Norm]  (Post-LN)"""

    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


# ══════════════════════════════════════════════════════════════════════
#   DECODER LAYER
# ══════════════════════════════════════════════════════════════════════


class DecoderLayer(nn.Module):
    """
    x → [Masked Self-Attn → Add & Norm]
      → [Cross-Attn(memory) → Add & Norm]
      → [FFN → Add & Norm]   (Post-LN)
    """

    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, memory, memory, src_mask)))
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x


# ══════════════════════════════════════════════════════════════════════
#   ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════


class Encoder(nn.Module):
    """Stack of N identical EncoderLayer modules with final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of N identical DecoderLayer modules with final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
#   FULL TRANSFORMER
# ══════════════════════════════════════════════════════════════════════


class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for German→English machine translation.

    All arguments carry defaults that match the trained checkpoint so the
    autograder can instantiate with Transformer() and call model.infer().
    Vocab, tokenizers, and pretrained weights are all loaded inside __init__.
    """

    def __init__(
        self,
        d_model: int = 256,
        N: int = 3,
        num_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.1,
        checkpoint_path: str = "best_checkpoint.pth",
        load_pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # ── 1. Tokenizers (regex — no external model required) ───────
        self._tok_de = _tok_de
        self._tok_en = _tok_en

        # ── 2. Download checkpoint if needed (inference / autograder) ─
        ckpt = None
        if load_pretrained:
            if GDRIVE_FILE_ID and not GDRIVE_FILE_ID.startswith("<"):
                if not os.path.exists(checkpoint_path):
                    gdown.download(id=GDRIVE_FILE_ID, output=checkpoint_path, quiet=False)
            if os.path.exists(checkpoint_path):
                ckpt = torch.load(checkpoint_path, map_location="cpu")

        # ── 3. Vocabulary — from checkpoint or built fresh ────────────
        if ckpt is not None and "src_vocab" in ckpt and "tgt_vocab" in ckpt:
            self._src_vocab = ckpt["src_vocab"]
            self._tgt_vocab = ckpt["tgt_vocab"]
        else:
            from dataset import Multi30kDataset
            _train = Multi30kDataset("train")
            self._src_vocab, self._tgt_vocab = _train.build_vocab(min_freq=2)

        src_vocab_size = len(self._src_vocab)
        tgt_vocab_size = len(self._tgt_vocab)

        # ── 4. Model architecture ────────────────────────────────────
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_enc   = PositionalEncoding(d_model, dropout)

        enc_layer    = EncoderLayer(d_model, num_heads, d_ff, dropout)
        dec_layer    = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder = Encoder(enc_layer, N)
        self.decoder = Decoder(dec_layer, N)
        self.fc_out  = nn.Linear(d_model, tgt_vocab_size)

        self._init_weights()

        # ── 5. Load pretrained weights ────────────────────────────────
        if ckpt is not None:
            self.load_state_dict(ckpt.get("model_state_dict", ckpt))
            print(f"Loaded pretrained weights from '{checkpoint_path}'")

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ── AUTOGRADER HOOKS ────────────────────────────────────────────

    def encode(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            src      : [batch, src_len]
            src_mask : [batch, 1, 1, src_len]
        Returns:
            memory : [batch, src_len, d_model]
        """
        x = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        return self.encoder(x, src_mask)

    def decode(
        self,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            memory   : [batch, src_len, d_model]
            src_mask : [batch, 1, 1, src_len]
            tgt      : [batch, tgt_len]
            tgt_mask : [batch, 1, tgt_len, tgt_len]
        Returns:
            logits : [batch, tgt_len, tgt_vocab_size]
        """
        x = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        x = self.decoder(x, memory, src_mask, tgt_mask)
        return self.fc_out(x)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            src      : [batch, src_len]
            tgt      : [batch, tgt_len]
            src_mask : [batch, 1, 1, src_len]
            tgt_mask : [batch, 1, tgt_len, tgt_len]
        Returns:
            logits : [batch, tgt_len, tgt_vocab_size]
        """
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    def infer(self, src_sentence: str) -> str:
        """
        Translates a German sentence to English using greedy autoregressive decoding.

        Args:
            src_sentence : Raw German text.
        Returns:
            Translated English string.
        """
        self.eval()
        device = next(self.parameters()).device

        src_stoi = self._src_vocab.stoi
        tgt_stoi = self._tgt_vocab.stoi
        sos_src = src_stoi["<sos>"]
        eos_src = src_stoi["<eos>"]
        sos_tgt = tgt_stoi["<sos>"]
        eos_tgt = tgt_stoi["<eos>"]
        pad_tgt = tgt_stoi["<pad>"]
        unk = src_stoi["<unk>"]

        tokens = self._tok_de(src_sentence)
        ids = [sos_src] + [src_stoi.get(t, unk) for t in tokens] + [eos_src]
        src = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        src_mask = make_src_mask(src)

        with torch.no_grad():
            memory = self.encode(src, src_mask)
            ys = torch.tensor([[sos_tgt]], dtype=torch.long, device=device)
            for _ in range(100):
                tgt_mask = make_tgt_mask(ys)
                logits = self.decode(memory, src_mask, ys, tgt_mask)
                next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                ys = torch.cat([ys, next_tok], dim=1)
                if next_tok.item() == eos_tgt:
                    break

        out = []
        for idx in ys[0].tolist():
            if idx == eos_tgt:
                break
            if idx not in (sos_tgt, pad_tgt):
                out.append(self._tgt_vocab.lookup_token(idx))
        return " ".join(out)
