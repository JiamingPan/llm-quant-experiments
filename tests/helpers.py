from types import SimpleNamespace

import torch
import torch.nn as nn


class ToyTokenizer:
    pad_token_id = 0

    def __call__(
        self,
        texts,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=512,
        add_special_tokens=True,
    ):
        if isinstance(texts, str):
            texts = [texts]

        rows = []
        for text in texts:
            ids = [1] if add_special_tokens else []
            ids.extend([(ord(ch) % 17) + 2 for ch in text])
            if truncation:
                ids = ids[:max_length]
            if len(ids) < 2:
                ids.append(2)
            rows.append(ids)

        width = max(len(row) for row in rows) if padding else None
        padded = []
        masks = []
        for row in rows:
            if width is None:
                padded.append(row)
                masks.append([1] * len(row))
            else:
                pad = width - len(row)
                padded.append(row + [self.pad_token_id] * pad)
                masks.append([1] * len(row) + [0] * pad)

        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }


class ToyBlock(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        return x + torch.tanh(self.proj(x))


class ToyInner(nn.Module):
    def __init__(self, hidden, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([ToyBlock(hidden) for _ in range(n_layers)])


class ToyLM(nn.Module):
    def __init__(self, vocab=32, hidden=8, n_layers=2):
        super().__init__()
        torch.manual_seed(0)
        self.embed = nn.Embedding(vocab, hidden)
        self.model = ToyInner(hidden, n_layers)
        self.head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids, attention_mask=None, output_attentions=False):
        hidden = self.embed(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)
        logits = self.head(hidden)
        if output_attentions:
            batch, seq_len = input_ids.shape
            attentions = tuple(
                torch.full((batch, 2, seq_len, seq_len), 1.0 / seq_len, device=input_ids.device)
                for _ in self.model.layers
            )
            return SimpleNamespace(logits=logits, attentions=attentions)
        return SimpleNamespace(logits=logits)
