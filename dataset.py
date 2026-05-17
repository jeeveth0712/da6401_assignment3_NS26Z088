import re
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from collections import Counter


def _make_tokenizer(model_name: str):
    try:
        import spacy as _spacy  # noqa: PLC0415
        _nlp = _spacy.load(model_name)
        return lambda text: [t.text.lower() for t in _nlp(text)]
    except OSError:
        return lambda text: re.findall(r"\w+|[^\w\s]", text.lower())


# ══════════════════════════════════════════════════════════════════════
#   VOCABULARY
# ══════════════════════════════════════════════════════════════════════

class Vocabulary:
    """Simple word-level vocabulary with special tokens."""

    SPECIALS = ['<unk>', '<pad>', '<sos>', '<eos>']

    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.stoi = {tok: i for i, tok in enumerate(self.SPECIALS)}
        self.itos = {i: tok for i, tok in enumerate(self.SPECIALS)}

    def build(self, sentences, tokenizer_fn):
        counter = Counter()
        for sent in sentences:
            counter.update(tokenizer_fn(sent))
        for word, freq in counter.items():
            if freq >= self.min_freq and word not in self.stoi:
                idx = len(self.stoi)
                self.stoi[word] = idx
                self.itos[idx] = word

    def __len__(self):
        return len(self.stoi)

    def lookup_token(self, idx: int) -> str:
        return self.itos.get(idx, '<unk>')

    def encode(self, tokens) -> list:
        unk = self.stoi['<unk>']
        return [self.stoi.get(t, unk) for t in tokens]


# ══════════════════════════════════════════════════════════════════════
#   DATASET
# ══════════════════════════════════════════════════════════════════════

class Multi30kDataset(Dataset):
    """
    Wraps the bentrevett/multi30k HuggingFace dataset.

    Usage
    -----
    train_ds = Multi30kDataset('train')
    src_vocab, tgt_vocab = train_ds.build_vocab()
    train_ds.process_data()

    val_ds = Multi30kDataset('validation')
    val_ds.set_vocab(src_vocab, tgt_vocab)
    val_ds.process_data()
    """

    def __init__(self, split: str = 'train'):
        self.split = split
        self._tok_de = _make_tokenizer("de_core_news_sm")
        self._tok_en = _make_tokenizer("en_core_web_sm")

        raw = load_dataset('bentrevett/multi30k')
        self.raw_data = raw[split]

        self.src_vocab = None
        self.tgt_vocab = None
        self.data      = []

    # ── vocab ────────────────────────────────────────────────────────

    def build_vocab(self, min_freq: int = 2):
        """Build src (de) and tgt (en) vocabularies from this split's data."""
        self.src_vocab = Vocabulary(min_freq)
        self.tgt_vocab = Vocabulary(min_freq)
        self.src_vocab.build([item['de'] for item in self.raw_data], self._tok_de)
        self.tgt_vocab.build([item['en'] for item in self.raw_data], self._tok_en)
        return self.src_vocab, self.tgt_vocab

    def set_vocab(self, src_vocab: Vocabulary, tgt_vocab: Vocabulary):
        """Share pre-built vocabularies (for val/test splits)."""
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    # ── processing ───────────────────────────────────────────────────

    def process_data(self):
        """Tokenize and convert to integer tensors. Must call set_vocab / build_vocab first."""
        assert self.src_vocab is not None, "Call build_vocab() or set_vocab() first."
        sos_s = self.src_vocab.stoi['<sos>']
        eos_s = self.src_vocab.stoi['<eos>']
        sos_t = self.tgt_vocab.stoi['<sos>']
        eos_t = self.tgt_vocab.stoi['<eos>']

        self.data = []
        for item in self.raw_data:
            src_ids = [sos_s] + self.src_vocab.encode(self._tok_de(item['de'])) + [eos_s]
            tgt_ids = [sos_t] + self.tgt_vocab.encode(self._tok_en(item['en'])) + [eos_t]
            self.data.append((
                torch.tensor(src_ids, dtype=torch.long),
                torch.tensor(tgt_ids, dtype=torch.long),
            ))
        return self.data

    # ── Dataset interface ────────────────────────────────────────────

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ══════════════════════════════════════════════════════════════════════
#   COLLATE FUNCTION  (pass to DataLoader)
# ══════════════════════════════════════════════════════════════════════

def collate_fn(batch, pad_idx: int = 1):
    """Pad variable-length sequences in a batch to equal length."""
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)
    return src_padded, tgt_padded
