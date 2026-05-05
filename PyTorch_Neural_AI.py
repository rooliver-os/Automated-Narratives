"""
train_davidson_narrative_ai.py

PyTorch/Hugging Face training script for Davidson automated basketball narratives.

Input:
    Full_Training_Data.csv  OR Full_Training_data.csv

The script uses the existing rightmost target column:
    Generated_Narrative_Target

It builds the model input directly from the existing statistical columns, so you do
NOT need to create a separate model-training CSV.

Run:
    pip install torch transformers sentencepiece pandas scikit-learn accelerate
    python train_davidson_narrative_ai.py --csv Full_Training_Data.csv

Output:
    davidson_narrative_ai/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


DEFAULT_MODEL_NAME = "google/flan-t5-small"
TARGET_COLUMN = "Generated_Narrative_Target"
SAVE_DIR = "davidson_narrative_ai"

# Keep this intentionally explicit. These are the facts the model sees.
FEATURE_COLUMNS = [
    "Date", "Season", "Opponent", "Home_Away_Neutral", "Game_Type", "Location",
    "Davidson_Record_After", "Opponent_Record_After",
    "Davidson_Final_Points", "Opponent_Final_Points", "Result", "Point_Differential",
    "Overtime", "Overtime_Periods",
    "Davidson_First_Half_Points", "Opponent_First_Half_Points",
    "Davidson_Second_Half_Points", "Opponent_Second_Half_Points",
    "Halftime_Leader", "Halftime_Margin", "Comeback_Win",
    "Lead_Changes", "Times_Tied", "Largest_Lead", "Largest_Lead_Team",
    "Davidson_FG_Made", "Davidson_FG_Attempted", "Davidson_FG_Percentage",
    "Opponent_FG_Made", "Opponent_FG_Attempted", "Opponent_FG_Percentage",
    "Davidson_3PT_Made", "Davidson_3PT_Attempted", "Davidson_3PT_Percentage",
    "Opponent_3PT_Made", "Opponent_3PT_Attempted", "Opponent_3PT_Percentage",
    "Davidson_FT_Made", "Davidson_FT_Attempted", "Davidson_FT_Percentage",
    "Opponent_FT_Made", "Opponent_FT_Attempted", "Opponent_FT_Percentage",
    "Davidson_Rebounds", "Opponent_Rebounds", "Rebound_Differential",
    "Davidson_Assists", "Opponent_Assists",
    "Davidson_Turnovers", "Opponent_Turnovers", "Turnover_Differential",
    "Davidson_Steals", "Opponent_Steals", "Davidson_Blocks", "Opponent_Blocks",
    "Davidson_Bench_Points", "Opponent_Bench_Points",
    "Davidson_Points_In_Paint", "Opponent_Points_In_Paint",
    "Davidson_Fast_Break_Points", "Opponent_Fast_Break_Points",
    "Davidson_Second_Chance_Points", "Opponent_Second_Chance_Points",
    "Davidson_Points_Off_Turnovers", "Opponent_Points_Off_Turnovers",
    "Key_Run_1", "Key_Run_2", "Game_Winning_Shot",
    "Top_Scorer_Davidson", "Top_Scorer_Davidson_Points",
    "Top_Scorer_Davidson_Rebounds", "Top_Scorer_Davidson_Assists", "Top_Scorer_Davidson_Minutes",
    "Second_Key_Player_Davidson", "Second_Key_Player_Davidson_Statline",
    "Third_Key_Player_Davidson", "Third_Key_Player_Davidson_Statline",
    "Top_Scorer_Opponent", "Top_Scorer_Opponent_Points",
    "Top_Scorer_Opponent_Rebounds", "Top_Scorer_Opponent_Assists",
    "Double_Doubles_Davidson", "Triple_Doubles_Davidson",
]


def clean_value(value: Any) -> str:
    """Convert missing values into a clean blank string."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def build_model_input(row: pd.Series) -> str:
    """
    Turn one game row into the prompt/input that the model trains on.
    The target is still row[Generated_Narrative_Target].
    """
    lines = [
        "Task: Generate a factual two-paragraph Davidson basketball game recap.",
        "Use only the facts given below. Do not invent player statistics.",
        "",
        "Game facts:",
    ]

    for col in FEATURE_COLUMNS:
        if col in row.index:
            value = clean_value(row[col])
            if value != "":
                pretty_col = col.replace("_", " ")
                lines.append(f"{pretty_col}: {value}")

    return "\n".join(lines)


class DavidsonNarrativeDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_input_length: int, max_target_length: int):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        source_text = build_model_input(row)
        target_text = clean_value(row[TARGET_COLUMN])

        source = self.tokenizer(
            source_text,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        target = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = target["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source["input_ids"].squeeze(0),
            "attention_mask": source["attention_mask"].squeeze(0),
            "labels": labels,
        }


def find_default_csv() -> Path:
    candidates = [Path("Full_Training_Data.csv"), Path("Full_Training_data.csv"), Path("Full_Training_data_FIXED.csv")]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find Full_Training_Data.csv in this folder.")


def train(args: argparse.Namespace) -> None:
    csv_path = Path(args.csv) if args.csv else find_default_csv()
    df = pd.read_csv(csv_path)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing required target column: {TARGET_COLUMN}")

    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    df = df[df[TARGET_COLUMN].astype(str).str.strip() != ""].copy()

    if len(df) < 10:
        raise ValueError("This training file is very small. Add more games before training.")

    # With small data, validation has to be small too. This still lets us monitor loss.
    train_df, val_df = train_test_split(df, test_size=args.val_size, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    train_dataset = DavidsonNarrativeDataset(train_df, tokenizer, args.max_input_length, args.max_target_length)
    val_dataset = DavidsonNarrativeDataset(val_df, tokenizer, args.max_input_length, args.max_target_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    print(f"Training on {len(train_df)} games; validating on {len(val_df)} games.")
    print(f"Device: {device}")
    print(f"Base model: {args.model_name}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / max(len(train_loader), 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / max(len(val_loader), 1)
        print(f"Epoch {epoch}/{args.epochs} | train loss: {avg_train_loss:.4f} | val loss: {avg_val_loss:.4f}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    # Save the feature list so generation uses the same structure.
    (save_dir / "feature_columns.txt").write_text("\n".join(FEATURE_COLUMNS), encoding="utf-8")
    print(f"Saved trained model to {save_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None, help="Path to Full_Training_Data.csv")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--save-dir", default=SAVE_DIR)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-input-length", type=int, default=768)
    parser.add_argument("--max-target-length", type=int, default=320)
    parser.add_argument("--val-size", type=float, default=0.20)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
