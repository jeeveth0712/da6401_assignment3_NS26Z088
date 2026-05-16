"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from functools import partial
from typing import Optional

import wandb

from model import Transformer, make_src_mask, make_tgt_mask
from lr_scheduler import NoamScheduler
from dataset import Multi30kDataset, collate_fn


# ══════════════════════════════════════════════════════════════════════
#   LABEL SMOOTHING LOSS
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in §5.4:
        y_smooth = (1 - ε) · one_hot(y) + ε / (V - 1)  for non-pad positions
        PAD token always receives 0 probability mass.

    Args:
        vocab_size : Number of output classes.
        pad_idx    : Index of <pad> token.
        smoothing  : Smoothing factor ε (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : [N, vocab_size]  (raw logits, before softmax)
            target : [N]              (gold token indices)
        Returns:
            Scalar loss.
        """
        log_probs = F.log_softmax(logits, dim=-1)

        # smooth distribution: ε/(V-1) everywhere, (1-ε) on gold token
        smooth_val  = self.smoothing / (self.vocab_size - 1)
        target_dist = torch.full_like(log_probs, smooth_val)
        target_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        target_dist[:, self.pad_idx] = 0.0

        # zero loss for padding positions
        pad_mask = (target == self.pad_idx)
        loss = -(target_dist * log_probs).sum(dim=-1)
        loss = loss.masked_fill(pad_mask, 0.0)

        n_tokens = (~pad_mask).sum().clamp(min=1).float()
        return loss.sum() / n_tokens


# ══════════════════════════════════════════════════════════════════════
#   TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.

    Returns:
        avg_loss : Average per-token loss over the epoch.
    """
    model.train() if is_train else model.eval()
    total_loss = 0.0
    n_batches  = 0

    with torch.set_grad_enabled(is_train):
        for src, tgt in data_iter:
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_in  = tgt[:, :-1]   # decoder input  (drop last token)
            tgt_out = tgt[:, 1:]    # expected output (drop <sos>)

            src_mask = make_src_mask(src)
            tgt_mask = make_tgt_mask(tgt_in)

            logits = model(src, tgt_in, src_mask, tgt_mask)

            logits_flat = logits.contiguous().view(-1, logits.size(-1))
            tgt_flat    = tgt_out.contiguous().view(-1)

            loss = loss_fn(logits_flat, tgt_flat)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            total_loss += loss.item()
            n_batches  += 1

    return total_loss / n_batches if n_batches > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════
#   GREEDY DECODING
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : [1, src_len]
        src_mask     : [1, 1, 1, src_len]
        max_len      : Maximum tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.
    Returns:
        ys : [1, out_len]  includes start_symbol; stops at end_symbol or max_len.
    """
    model.eval()
    with torch.no_grad():
        memory = model.encode(src, src_mask)
        ys = torch.full((1, 1), start_symbol, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys)
            logits   = model.decode(memory, src_mask, ys, tgt_mask)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_tok], dim=1)
            if next_tok.item() == end_symbol:
                break

    return ys


# ══════════════════════════════════════════════════════════════════════
#   BLEU EVALUATION
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0–100).
    """
    try:
        from sacrebleu.metrics import BLEU as SacreBLEU
        bleu_scorer = SacreBLEU(tokenize='none')
        use_sacrebleu = True
    except ImportError:
        use_sacrebleu = False

    pad_idx = tgt_vocab.stoi.get('<pad>', 1)
    sos_idx = tgt_vocab.stoi.get('<sos>', 2)
    eos_idx = tgt_vocab.stoi.get('<eos>', 3)

    model.eval()
    hypotheses = []
    references = []

    with torch.no_grad():
        for src, tgt in test_dataloader:
            src = src.to(device)
            for i in range(src.size(0)):
                src_i    = src[i].unsqueeze(0)
                src_mask = make_src_mask(src_i)
                pred     = greedy_decode(
                    model, src_i, src_mask, max_len, sos_idx, eos_idx, device
                )

                hyp_ids = pred[0].tolist()
                hyp = []
                for idx in hyp_ids:
                    if idx == eos_idx:
                        break
                    if idx not in (sos_idx, pad_idx):
                        hyp.append(tgt_vocab.lookup_token(idx))

                ref_ids = tgt[i].tolist()
                ref = []
                for idx in ref_ids:
                    if idx == eos_idx:
                        break
                    if idx not in (sos_idx, pad_idx):
                        ref.append(tgt_vocab.lookup_token(idx))

                hypotheses.append(' '.join(hyp))
                references.append(' '.join(ref))

    if use_sacrebleu:
        result = bleu_scorer.corpus_score(hypotheses, [references])
        return float(result.score)

    # fallback: use torchtext bleu_score (returns 0–1, multiply by 100)
    from torchtext.data.metrics import bleu_score as tt_bleu
    hyp_tokens = [h.split() for h in hypotheses]
    ref_tokens = [[r.split()] for r in references]
    return tt_bleu(hyp_tokens, ref_tokens) * 100.0


# ══════════════════════════════════════════════════════════════════════
#   CHECKPOINT UTILITIES
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """Save model + optimizer + scheduler state."""
    enc_layer = model.encoder.layers[0]
    model_config = {
        'd_model':   model.d_model,
        'N':         len(model.encoder.layers),
        'num_heads': enc_layer.self_attn.num_heads,
        'd_ff':      enc_layer.ffn.linear1.out_features,
        'dropout':   enc_layer.dropout.p,
    }
    torch.save({
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'model_config':         model_config,
        'src_vocab':            model._src_vocab,
        'tgt_vocab':            model._tgt_vocab,
    }, path)


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """Restore model (and optionally optimizer/scheduler) from disk."""
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch']


# ══════════════════════════════════════════════════════════════════════
#   EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment() -> None:
    """Full training experiment with W&B logging."""

    config = dict(
        d_model      = 256,
        N            = 3,
        num_heads    = 8,
        d_ff         = 512,
        dropout      = 0.1,
        batch_size   = 128,
        num_epochs   = 15,
        warmup_steps = 4000,
        label_smooth = 0.1,
        max_len      = 100,
    )

    wandb.init(project="da6401-a3", config=config)
    cfg = wandb.config

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── model (vocab + tokenizers built inside __init__) ─────────────
    # GDRIVE_FILE_ID is a placeholder during training, so no weights load.
    model = Transformer(
        d_model   = cfg.d_model,
        N         = cfg.N,
        num_heads = cfg.num_heads,
        d_ff      = cfg.d_ff,
        dropout   = cfg.dropout,
    ).to(device)

    # Reuse vocab that was built inside the model
    src_vocab = model._src_vocab
    tgt_vocab = model._tgt_vocab
    pad_idx   = src_vocab.stoi['<pad>']
    _collate  = partial(collate_fn, pad_idx=pad_idx)

    # ── datasets ─────────────────────────────────────────────────────
    train_ds = Multi30kDataset('train')
    train_ds.set_vocab(src_vocab, tgt_vocab)
    train_ds.process_data()

    val_ds = Multi30kDataset('validation')
    val_ds.set_vocab(src_vocab, tgt_vocab)
    val_ds.process_data()

    test_ds = Multi30kDataset('test')
    test_ds.set_vocab(src_vocab, tgt_vocab)
    test_ds.process_data()

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True,  collate_fn=_collate)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size,
                              shuffle=False, collate_fn=_collate)
    test_loader  = DataLoader(test_ds,  batch_size=1,
                              shuffle=False, collate_fn=_collate)

    # lr=1.0 because NoamScheduler scales base_lr × Noam_factor
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = NoamScheduler(optimizer, cfg.d_model, warmup_steps=cfg.warmup_steps)
    loss_fn   = LabelSmoothingLoss(len(tgt_vocab), pad_idx, smoothing=cfg.label_smooth)

    # ── training loop ────────────────────────────────────────────────
    best_val_loss = float('inf')
    for epoch in range(cfg.num_epochs):
        train_loss = run_epoch(
            train_loader, model, loss_fn, optimizer, scheduler,
            epoch_num=epoch, is_train=True, device=device
        )
        val_loss = run_epoch(
            val_loader, model, loss_fn, None, None,
            epoch_num=epoch, is_train=False, device=device
        )

        wandb.log({
            'epoch':      epoch + 1,
            'train_loss': train_loss,
            'val_loss':   val_loss,
            'lr':         scheduler.get_last_lr()[0],
        })

        print(f"Epoch {epoch+1:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, path="best_checkpoint.pth")

    # ── final BLEU ───────────────────────────────────────────────────
    load_checkpoint("best_checkpoint.pth", model)
    bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device, max_len=cfg.max_len)
    print(f"Test BLEU: {bleu:.2f}")
    wandb.log({'test_bleu': bleu})
    wandb.finish()


if __name__ == "__main__":
    run_training_experiment()
