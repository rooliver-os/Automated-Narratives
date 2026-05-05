"""
generate_davidson_narrative.py

Loads the trained Davidson narrative AI and generates a two-paragraph recap.

Examples:
    python generate_davidson_narrative.py --row 0 --csv Full_Training_Data.csv
    python generate_davidson_narrative.py --json new_game.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_DIR = "davidson_narrative_ai"

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
    if pd.isna(value):
        return ""
    return str(value).strip()


def build_model_input(row: pd.Series | dict[str, Any]) -> str:
    lines = [
        "Task: Generate a factual two-paragraph Davidson basketball game recap.",
        "Use only the facts given below. Do not invent player statistics.",
        "",
        "Game facts:",
    ]

    for col in FEATURE_COLUMNS:
        if col in row:
            value = clean_value(row[col])
            if value != "":
                lines.append(f"{col.replace('_', ' ')}: {value}")

    return "\n".join(lines)


def load_input(args: argparse.Namespace) -> pd.Series | dict[str, Any]:
    if args.json:
        return json.loads(Path(args.json).read_text(encoding="utf-8"))

    if args.csv is None:
        raise ValueError("Provide either --csv and --row, or --json.")

    df = pd.read_csv(args.csv)
    if args.row < 0 or args.row >= len(df):
        raise IndexError(f"Row {args.row} is out of range for {args.csv}.")
    return df.iloc[args.row]


def generate(args: argparse.Namespace) -> None:
    model_dir = Path(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()

    row = load_input(args)
    input_text = build_model_input(row)

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=args.max_input_length,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=args.max_output_length,
            num_beams=args.num_beams,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

    narrative = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(narrative)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--json", default=None, help="Optional JSON file containing one new game's stats.")
    parser.add_argument("--max-input-length", type=int, default=768)
    parser.add_argument("--max-output-length", type=int, default=320)
    parser.add_argument("--num-beams", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
