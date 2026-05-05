"""
PyTorch model for generating basketball game narratives.


* **Dropout and LayerNorm** in the statistical encoder to mitigate
  overfitting and stabilise training.
* A **multi‑layer LSTM decoder** with dropout between layers and
  embedding dropout for better generalisation.
* **Teacher forcing with scheduled sampling**: gradually reduces
  reliance on ground‑truth tokens during training.  You can set the
  initial ratio and decay rate via command line arguments.
* **Gradient clipping** to prevent exploding gradients.
* **Early stopping** based on validation loss to avoid overfitting.
* A simple **beam search** implementation for inference, which
  generates higher quality narratives compared to greedy decoding.

To train the model run:

    python narrative_model_tuned.py --train_csv train_dataset.csv \
        --val_csv validation_dataset.csv --epochs 30 --batch_size 16 \
        --teacher_forcing_start 1.0 --teacher_forcing_end 0.5 --teacher_forcing_decay 0.95

"""

import argparse
import random
import math
import os
from typing import List, Tuple, Dict, Any

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# ---------------------------- Utility functions -----------------------------

def tokenize(text: str) -> List[str]:
    """A slightly smarter tokenizer splitting on whitespace and punctuation."""
    import re
    # Split on word boundaries and keep punctuation as separate tokens
    tokens = re.findall(r"\w+|[^\w\s]", text.lower())
    return tokens

class Vocab:
    """Token vocabulary with special tokens and frequency threshold."""

    def __init__(self, min_freq: int = 1):
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.min_freq = min_freq
        # Define special tokens
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"
        for tok in [self.pad_token, self.bos_token, self.eos_token, self.unk_token]:
            self._add_token(tok)

    def _add_token(self, token: str) -> int:
        idx = len(self.token_to_id)
        self.token_to_id[token] = idx
        self.id_to_token[idx] = token
        return idx

    def build_from_texts(self, texts: List[str]) -> None:
        from collections import Counter
        counter = Counter()
        for text in texts:
            counter.update(tokenize(text))
        for token, freq in counter.items():
            if freq >= self.min_freq and token not in self.token_to_id:
                self._add_token(token)

    def __len__(self) -> int:
        return len(self.token_to_id)

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.token_to_id.get(tok, self.token_to_id[self.unk_token]) for tok in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        return [self.id_to_token.get(idx, self.unk_token) for idx in ids]

# ------------------------------ Dataset class -------------------------------

class NarrativeDataset(Dataset):
    """Dataset storing game statistics and narrative sequences."""

    def __init__(self, dataframe: pd.DataFrame, feature_columns: List[str], vocab: Vocab):
        self.vocab = vocab
        self.features = dataframe[feature_columns].astype(float).values
        self.sequences = []
        for text in dataframe["Generated_Narrative_Target"].astype(str).tolist():
            tokens = [vocab.bos_token] + tokenize(text) + [vocab.eos_token]
            self.sequences.append(torch.tensor(vocab.encode(tokens), dtype=torch.long))

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.features[idx], dtype=torch.float32), self.sequences[idx]


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    feature_list, seq_list = zip(*batch)
    features = torch.stack(feature_list, dim=0)
    input_seqs = [s[:-1] for s in seq_list]
    target_seqs = [s[1:] for s in seq_list]
    input_padded = pad_sequence(input_seqs, batch_first=True, padding_value=0)
    target_padded = pad_sequence(target_seqs, batch_first=True, padding_value=0)
    return features, input_padded, target_padded

# ----------------------------- Model definitions ----------------------------

class StatEncoder(nn.Module):
    """Enhanced statistical encoder with normalization and dropout."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = []
        current_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 2 * hidden_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.network(x)
        hidden_dim = out.size(1) // 2
        h0 = out[:, :hidden_dim].unsqueeze(0)
        c0 = out[:, hidden_dim:].unsqueeze(0)
        return h0, c0

class NarrativeDecoder(nn.Module):
    """Multi‑layer LSTM decoder with embedding and output dropout."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.output_dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_tokens: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        embeds = self.embedding(input_tokens)
        embeds = self.embed_dropout(embeds)
        output, hidden = self.lstm(embeds, hidden)
        output = self.output_dropout(output)
        logits = self.out(output)
        return logits, hidden

class Seq2Seq(nn.Module):
    """Sequence‑to‑sequence model combining encoder and decoder with scheduled sampling."""

    def __init__(self, input_dim: int, vocab_size: int, embed_dim: int = 128,
                 hidden_dim: int = 256, encoder_layers: int = 2, decoder_layers: int = 2,
                 encoder_dropout: float = 0.1, decoder_dropout: float = 0.3):
        super().__init__()
        self.encoder = StatEncoder(input_dim, hidden_dim, num_layers=encoder_layers, dropout=encoder_dropout)
        self.decoder = NarrativeDecoder(vocab_size, embed_dim, hidden_dim, num_layers=decoder_layers, dropout=decoder_dropout)
        self.vocab_size = vocab_size

    def forward(self, stats: torch.Tensor, target: torch.Tensor, teacher_forcing_ratio: float) -> torch.Tensor:
        """Forward pass with teacher forcing.

        Args:
            stats: (batch, input_dim) numeric features
            target: (batch, seq_len) target token sequences including BOS/EOS
            teacher_forcing_ratio: probability of using ground truth token as next input
        Returns:
            logits: (batch, seq_len - 1, vocab_size)
        """
        batch_size, seq_len = target.size()
        h0, c0 = self.encoder(stats)
        hidden = (h0, c0)
        # Prepare input and output containers
        input_token = target[:, 0].unsqueeze(1)  # start with BOS
        outputs = []
        for t in range(1, seq_len):
            logits, hidden = self.decoder(input_token, hidden)
            logits_step = logits  # shape: (batch, 1, vocab_size)
            outputs.append(logits_step)
            # Choose next input token: teacher forcing or predicted
            use_teacher = random.random() < teacher_forcing_ratio
            if use_teacher:
                input_token = target[:, t].unsqueeze(1)
            else:
                next_tokens = logits_step.argmax(-1)
                input_token = next_tokens
        return torch.cat(outputs, dim=1)

    @torch.no_grad()
    def generate(self, stats: torch.Tensor, max_len: int, vocab: Vocab, beam_size: int = 3, device: str = "cpu") -> str:
        """Generate narrative using beam search.

        Args:
            stats: (input_dim,) numeric features
            max_len: maximum generated length
            vocab: vocabulary for decoding
            beam_size: number of beams
        Returns:
            Generated narrative as a single string
        """
        self.eval()
        stats = stats.unsqueeze(0).to(device)
        h0, c0 = self.encoder(stats)
        # Each beam stores (tokens, hidden (h,c), score)
        beams = [([vocab.token_to_id[vocab.bos_token]], (h0, c0), 0.0)]
        completed = []
        for _ in range(max_len):
            new_beams = []
            for tokens, hidden, score in beams:
                last_token = torch.tensor([[tokens[-1]]], device=device)
                logits, hidden_new = self.decoder(last_token, hidden)
                log_probs = torch.log_softmax(logits.squeeze(1), dim=-1)
                topk_scores, topk_tokens = torch.topk(log_probs, beam_size, dim=-1)
                for i in range(beam_size):
                    token_id = topk_tokens[i].item()
                    token_score = topk_scores[i].item()
                    new_seq = tokens + [token_id]
                    new_score = score + token_score
                    new_hidden = (hidden_new[0].clone(), hidden_new[1].clone())
                    new_beams.append((new_seq, new_hidden, new_score))
            # Keep best beams
            new_beams.sort(key=lambda x: x[2], reverse=True)
            beams = new_beams[:beam_size]
            # Check for completed sequences
            beams_to_remove = []
            for tokens, hidden, score in beams:
                if tokens[-1] == vocab.token_to_id[vocab.eos_token]:
                    completed.append((tokens, score))
                    beams_to_remove.append((tokens, hidden, score))
            beams = [b for b in beams if b not in beams_to_remove]
            if not beams:
                break
        if not completed:
            # If nothing completed, take highest scoring beam
            completed = [(beams[0][0], beams[0][2])]
        # Select best completed sequence
        completed.sort(key=lambda x: x[1], reverse=True)
        best_tokens = completed[0][0]
        # Remove BOS and EOS tokens
        decoded_tokens = [vocab.id_to_token[id_] for id_ in best_tokens[1:]]
        if decoded_tokens and decoded_tokens[-1] == vocab.eos_token:
            decoded_tokens = decoded_tokens[:-1]
        return " ".join(decoded_tokens)

# ------------------------------- Training loop ------------------------------

def train_model(model: Seq2Seq, train_loader: DataLoader, val_loader: DataLoader, vocab: Vocab,
                epochs: int, lr: float, device: str, tf_start: float, tf_end: float, tf_decay: float,
                grad_clip: float, patience: int) -> None:
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    teacher_forcing_ratio = tf_start
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        # Decay teacher forcing ratio
        teacher_forcing_ratio = max(tf_end, teacher_forcing_ratio * tf_decay)
        for stats, input_seqs, target_seqs in train_loader:
            stats = stats.to(device)
            input_seqs = input_seqs.to(device)
            target_seqs = target_seqs.to(device)
            optimizer.zero_grad()
            logits = model(stats, torch.cat([input_seqs[:, :1], target_seqs], dim=1), teacher_forcing_ratio)
            # logits shape (batch, seq_len, vocab_size), target shape (batch, seq_len)
            loss = criterion(logits.view(-1, logits.size(-1)), target_seqs.view(-1))
            loss.backward()
            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:2d} | train loss {avg_loss:.4f} | val loss {val_loss:.4f} | TF ratio {teacher_forcing_ratio:.2f}")
        # Early stopping
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab": vocab.token_to_id
            }, "best_narrative_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Stopping early due to no validation improvement.")
                break

@torch.no_grad()
def evaluate(model: Seq2Seq, data_loader: DataLoader, device: str) -> float:
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    total_loss = 0.0
    for stats, input_seqs, target_seqs in data_loader:
        stats = stats.to(device)
        input_seqs = input_seqs.to(device)
        target_seqs = target_seqs.to(device)
        logits = model(stats, torch.cat([input_seqs[:, :1], target_seqs], dim=1), teacher_forcing_ratio=0.0)
        loss = criterion(logits.view(-1, logits.size(-1)), target_seqs.view(-1))
        total_loss += loss.item()
    return total_loss / len(data_loader)

# ------------------------------- Main entry --------------------------------

def prepare_dataloaders(train_csv: str, val_csv: str, feature_columns: List[str], batch_size: int) -> Tuple[DataLoader, DataLoader, Vocab]:
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    vocab = Vocab(min_freq=1)
    vocab.build_from_texts(train_df["Generated_Narrative_Target"].astype(str).tolist())
    train_dataset = NarrativeDataset(train_df, feature_columns, vocab)
    val_dataset = NarrativeDataset(val_df, feature_columns, vocab)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader, vocab

def select_numeric_columns(df: pd.DataFrame) -> List[str]:
    numeric_cols = []
    for col in df.columns:
        if col == "Generated_Narrative_Target":
            continue
        try:
            pd.to_numeric(df[col])
            numeric_cols.append(col)
        except ValueError:
            pass
    return numeric_cols

def main() -> None:
    parser = argparse.ArgumentParser(description="Train an advanced model to generate basketball game narratives.")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to the training CSV file")
    parser.add_argument("--val_csv", type=str, required=True, help="Path to the validation CSV file")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--embed_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension of the LSTM")
    parser.add_argument("--encoder_layers", type=int, default=2, help="Number of layers in encoder MLP")
    parser.add_argument("--decoder_layers", type=int, default=2, help="Number of layers in decoder LSTM")
    parser.add_argument("--encoder_dropout", type=float, default=0.1, help="Dropout rate in encoder")
    parser.add_argument("--decoder_dropout", type=float, default=0.3, help="Dropout rate in decoder")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--teacher_forcing_start", type=float, default=1.0, help="Initial teacher forcing ratio")
    parser.add_argument("--teacher_forcing_end", type=float, default=0.5, help="Minimum teacher forcing ratio")
    parser.add_argument("--teacher_forcing_decay", type=float, default=0.95, help="Decay factor for teacher forcing each epoch")
    parser.add_argument("--grad_clip", type=float, default=5.0, help="Gradient clipping value")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda")
    args = parser.parse_args()

    # Load full training data to determine numeric feature columns
    full_train_df = pd.read_csv(args.train_csv)
    numeric_cols = select_numeric_columns(full_train_df)
    if not numeric_cols:
        raise ValueError("No numeric columns found. Please specify feature columns manually.")

    train_loader, val_loader, vocab = prepare_dataloaders(args.train_csv, args.val_csv, numeric_cols, args.batch_size)
    model = Seq2Seq(input_dim=len(numeric_cols), vocab_size=len(vocab),
                    embed_dim=args.embed_dim, hidden_dim=args.hidden_dim,
                    encoder_layers=args.encoder_layers, decoder_layers=args.decoder_layers,
                    encoder_dropout=args.encoder_dropout, decoder_dropout=args.decoder_dropout)
    train_model(model, train_loader, val_loader, vocab, epochs=args.epochs, lr=args.learning_rate,
                device=args.device, tf_start=args.teacher_forcing_start,
                tf_end=args.teacher_forcing_end, tf_decay=args.teacher_forcing_decay,
                grad_clip=args.grad_clip, patience=args.patience)

    # After training, save final model
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab": vocab.token_to_id
    }, "final_narrative_model.pt")
    print("Training complete. Model saved to final_narrative_model.pt")

if __name__ == "__main__":
    main()
