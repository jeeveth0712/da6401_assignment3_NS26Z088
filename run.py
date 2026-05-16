"""
run.py — CLI entry point for training and inference.

Usage
-----
# Train with defaults (matches the architecture saved to GDrive):
    python run.py train

# Train with custom hyperparameters:
    python run.py train --d_model 512 --N 6 --num_heads 8 --d_ff 2048 \
                        --dropout 0.1 --batch_size 64 --num_epochs 20 \
                        --warmup_steps 4000 --label_smooth 0.1 \
                        --checkpoint best_checkpoint.pth \
                        --wandb_project da6401-a3

# Translate a single German sentence (loads weights automatically):
    python run.py infer --sentence "Ein Mann sitzt auf einer Bank."

# Kaggle: suppress wandb sync and point checkpoint to /kaggle/working/
    python run.py train --no_wandb --checkpoint /kaggle/working/best_checkpoint.pth
"""

import argparse
import sys


# ══════════════════════════════════════════════════════════════════════
#   ARGUMENT PARSER
# ══════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="DA6401 A3 — Transformer NMT (German → English)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── train ──────────────────────────────────────────────────────
    tr = sub.add_parser("train", help="Train the Transformer model.",
                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Architecture  (must match defaults in Transformer.__init__ for autograder)
    tr.add_argument("--d_model",      type=int,   default=256,
                    help="Model dimensionality.")
    tr.add_argument("--N",            type=int,   default=3,
                    help="Number of encoder/decoder layers.")
    tr.add_argument("--num_heads",    type=int,   default=8,
                    help="Number of attention heads.")
    tr.add_argument("--d_ff",         type=int,   default=512,
                    help="FFN inner dimensionality.")
    tr.add_argument("--dropout",      type=float, default=0.1,
                    help="Dropout probability.")

    # Training
    tr.add_argument("--batch_size",   type=int,   default=128)
    tr.add_argument("--num_epochs",   type=int,   default=15)
    tr.add_argument("--warmup_steps", type=int,   default=4000,
                    help="Noam scheduler warmup steps.")
    tr.add_argument("--label_smooth", type=float, default=0.1,
                    help="Label smoothing epsilon.")
    tr.add_argument("--max_len",      type=int,   default=100,
                    help="Max decode length for BLEU evaluation.")
    tr.add_argument("--min_freq",     type=int,   default=2,
                    help="Min token frequency to include in vocabulary.")

    # I/O
    tr.add_argument("--checkpoint",   type=str,   default="best_checkpoint.pth",
                    help="Path to save the best model checkpoint.")
    tr.add_argument("--wandb_project",type=str,   default="da6401-a3",
                    help="W&B project name.")
    tr.add_argument("--no_wandb",     action="store_true",
                    help="Disable W&B logging (useful on Kaggle without API key).")

    # ── infer ──────────────────────────────────────────────────────
    inf = sub.add_parser("infer", help="Translate a German sentence.",
                          formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    inf.add_argument("--sentence", type=str, required=True,
                     help="German sentence to translate.")
    inf.add_argument("--checkpoint", type=str, default="best_checkpoint.pth",
                     help="Local checkpoint path (used if GDRIVE_FILE_ID not set).")

    return parser


# ══════════════════════════════════════════════════════════════════════
#   TRAIN
# ══════════════════════════════════════════════════════════════════════

def cmd_train(args):
    import os
    import torch
    from functools import partial

    import wandb

    from model import Transformer
    from dataset import Multi30kDataset, collate_fn
    from lr_scheduler import NoamScheduler
    from train import (
        LabelSmoothingLoss, run_epoch,
        evaluate_bleu, save_checkpoint, load_checkpoint,
    )
    from torch.utils.data import DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── W&B ──────────────────────────────────────────────────────────
    cfg_dict = vars(args)
    if args.no_wandb:
        os.environ["WANDB_MODE"] = "disabled"
    wandb.init(project=args.wandb_project, config=cfg_dict)
    cfg = wandb.config

    # ── Model (vocab + tokenizer built inside __init__) ───────────────
    print("Building model and vocabulary...")
    model = Transformer(
        d_model   = args.d_model,
        N         = args.N,
        num_heads = args.num_heads,
        d_ff      = args.d_ff,
        dropout   = args.dropout,
        checkpoint_path = args.checkpoint,
    ).to(device)

    src_vocab = model._src_vocab
    tgt_vocab = model._tgt_vocab
    pad_idx   = src_vocab.stoi['<pad>']
    _collate  = partial(collate_fn, pad_idx=pad_idx)

    # ── Datasets ─────────────────────────────────────────────────────
    print("Loading datasets...")
    train_ds = Multi30kDataset('train')
    train_ds.set_vocab(src_vocab, tgt_vocab)
    train_ds.process_data()

    val_ds = Multi30kDataset('validation')
    val_ds.set_vocab(src_vocab, tgt_vocab)
    val_ds.process_data()

    test_ds = Multi30kDataset('test')
    test_ds.set_vocab(src_vocab, tgt_vocab)
    test_ds.process_data()

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  collate_fn=_collate)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, collate_fn=_collate)
    test_loader  = DataLoader(test_ds,  batch_size=1,
                              shuffle=False, collate_fn=_collate)

    print(f"Vocab sizes — src: {len(src_vocab)}, tgt: {len(tgt_vocab)}")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    # ── Optimizer / Scheduler / Loss ─────────────────────────────────
    # lr=1.0 because NoamScheduler scales base_lr × Noam_factor
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = NoamScheduler(optimizer, args.d_model, warmup_steps=args.warmup_steps)
    loss_fn   = LabelSmoothingLoss(len(tgt_vocab), pad_idx, smoothing=args.label_smooth)

    # ── Training loop ────────────────────────────────────────────────
    best_val_loss = float('inf')
    for epoch in range(args.num_epochs):
        train_loss = run_epoch(
            train_loader, model, loss_fn, optimizer, scheduler,
            epoch_num=epoch, is_train=True, device=device,
        )
        val_loss = run_epoch(
            val_loader, model, loss_fn, None, None,
            epoch_num=epoch, is_train=False, device=device,
        )

        wandb.log({
            'epoch':      epoch + 1,
            'train_loss': train_loss,
            'val_loss':   val_loss,
            'lr':         scheduler.get_last_lr()[0],
        })
        print(f"Epoch {epoch+1:03d}/{args.num_epochs} "
              f"| train_loss={train_loss:.4f} | val_loss={val_loss:.4f} "
              f"| lr={scheduler.get_last_lr()[0]:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, path=args.checkpoint)
            print(f"  ✓ Saved checkpoint to {args.checkpoint}")

    # ── Final BLEU ───────────────────────────────────────────────────
    print("\nEvaluating on test set...")
    load_checkpoint(args.checkpoint, model)
    bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device, max_len=args.max_len)
    print(f"Test BLEU: {bleu:.2f}")
    wandb.log({'test_bleu': bleu})
    wandb.finish()


# ══════════════════════════════════════════════════════════════════════
#   INFER
# ══════════════════════════════════════════════════════════════════════

def cmd_infer(args):
    import torch
    from model import Transformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading model...")
    model = Transformer(checkpoint_path=args.checkpoint).to(device)
    model.eval()

    translation = model.infer(args.sentence)
    print(f"\nDE: {args.sentence}")
    print(f"EN: {translation}")


# ══════════════════════════════════════════════════════════════════════
#   ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "infer":
        cmd_infer(args)
