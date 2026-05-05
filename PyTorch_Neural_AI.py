"""
Small-data PyTorch AI for Davidson basketball narratives.

This version is designed for about 100 games. It does NOT try to generate every
word with an LSTM. Instead, it trains a neural network to choose a narrative
plan from game statistics, then fills clean Davidson-positive templates with the
real stats.

Train:
    python PyTorch_Neural_AI.py --mode train --train_csv train_dataset.csv --val_csv validation_dataset.csv --epochs 100

Generate one validation narrative:
    python PyTorch_Neural_AI.py --mode generate --input_csv validation_dataset.csv --row random

Generate a full output CSV:
    python PyTorch_Neural_AI.py --mode generate_file --input_csv validation_dataset.csv --output_csv ai_generated_output.csv
"""

import argparse
import os
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

OPENING_TEMPLATES = [
    "Davidson delivered a strong performance against {opponent}, finishing with a {score} result.",
    "Davidson battled through a competitive matchup against {opponent}, with the game ending {score}.",
    "Davidson showed resilience against {opponent}, continuing to compete throughout the {score} contest.",
    "Davidson responded well in key stretches against {opponent}, with the final score ending {score}.",
    "Davidson put together a determined effort against {opponent}, closing the game at {score}.",
    "Davidson controlled important stretches against {opponent}, earning a {score} result.",
    "Davidson used a balanced effort to challenge {opponent}, with the matchup finishing {score}.",
    "Davidson stayed composed in a tight contest against {opponent}, with the final score at {score}.",
    "Davidson pushed {opponent} throughout the night, with the game finishing {score}.",
    "Davidson found strong moments on both ends against {opponent}, with the final score at {score}.",
    "Davidson competed with energy and purpose against {opponent}, closing the matchup at {score}.",
]

PLAYER_TEMPLATES = [
    "{player} led Davidson with {points} points, {rebounds} rebounds and {assists} assists.",
    "{player} paced the Wildcats with {points} points while adding {rebounds} rebounds and {assists} assists.",
    "{player} provided a major lift for Davidson, finishing with {points} points, {rebounds} rebounds and {assists} assists.",
    "{player} gave Davidson steady production with {points} points, {rebounds} rebounds and {assists} assists.",
    "{player} delivered an important performance for Davidson with {points} points, {rebounds} rebounds and {assists} assists.",
    "{player} sparked the Wildcats with {points} points, {rebounds} rebounds and {assists} assists.",
    "{player} carried a major share of the Davidson offense with {points} points, {rebounds} rebounds and {assists} assists.",
]

SUPPORT_TEMPLATES = [
    "{second_player} added important support with {second_statline}.",
    "{second_player} also contributed with {second_statline}.",
    "{second_player} helped strengthen the Davidson effort with {second_statline}.",
    "{second_player} chipped in with {second_statline}, giving Davidson another reliable option.",
    "{second_player} provided valuable secondary production with {second_statline}.",
]

FLOW_TEMPLATES = [
    "The Wildcats continued to answer important runs and stayed connected throughout the game.",
    "Davidson found positive stretches through effort, execution and continued competitiveness.",
    "Davidson made key adjustments after halftime and continued to push the pace.",
    "The game featured several momentum swings, and Davidson continued to respond.",
    "Davidson showed composure in late possessions and remained competitive down the stretch.",
    "Davidson pushed the game beyond regulation and competed deep into overtime.",
    "After trailing at halftime, Davidson responded with stronger execution in the second half.",
    "Davidson built early momentum and continued to play with energy through the final minutes.",
    "Davidson used a strong second-half response to change the feel of the game.",
    "Davidson continued to fight through difficult stretches and kept searching for answers.",
]

TEAM_STAT_TEMPLATES = [
    "Davidson shot {fg_pct}% from the field, showing efficient offensive execution.",
    "Davidson connected on {threes_made} of {threes_attempted} shots from three-point range.",
    "Davidson converted {ft_pct}% of its free throws, showing discipline at the line.",
    "Davidson moved the ball well and finished with {team_assists} assists.",
    "Davidson found success inside, scoring {paint_points} points in the paint.",
    "Davidson turned defense into offense with {points_off_turnovers} points off turnovers.",
    "Davidson received production from its bench, which contributed {bench_points} points.",
    "Davidson competed on the glass and finished with {team_rebounds} rebounds.",
    "Davidson applied defensive pressure and collected {steals} steals.",
    "Davidson protected the rim with {blocks} blocks.",
]

CLOSING_TEMPLATES = [
    "With the result, Davidson can build on the positive stretches from this performance.",
    "Davidson will look to carry the encouraging parts of this performance into its next matchup.",
    "The performance gives Davidson several positives to build on moving forward.",
    "Davidson showed enough strong moments to take useful momentum into the next game.",
    "The Wildcats will aim to build from this effort as the season continues.",
    "The result reflects Davidson's continued ability to compete and find meaningful production.",
    "Davidson demonstrated resilience and will look to keep improving from here.",
    "Davidson can take away several encouraging signs from the way it competed.",
]

OUTPUT_SIZES = {
    "opening": len(OPENING_TEMPLATES),
    "player": len(PLAYER_TEMPLATES),
    "support": len(SUPPORT_TEMPLATES),
    "flow": len(FLOW_TEMPLATES),
    "team_stat": len(TEAM_STAT_TEMPLATES),
    "closing": len(CLOSING_TEMPLATES),
}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def get_field(row: pd.Series, name: str, default: Any = "") -> Any:
    value = row.get(name, default)
    if pd.isna(value):
        return default
    return value


def get_numeric_feature_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for col in df.columns:
        if col in {"Generated_Narrative_Target", "AI_Generated_Narrative"}:
            continue
        try:
            pd.to_numeric(df[col])
            cols.append(col)
        except Exception:
            pass
    return cols


def fit_scaler(df: pd.DataFrame, feature_columns: List[str]) -> Dict[str, List[float]]:
    X_df = df.reindex(columns=feature_columns, fill_value=0.0)
    X = X_df.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float).values
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return {"mean": mean.tolist(), "std": std.tolist()}


def transform_features(df: pd.DataFrame, feature_columns: List[str], scaler: Dict[str, List[float]]) -> np.ndarray:
    # Important fix: custom test CSVs can omit many training columns.
    # Missing numeric columns are automatically filled with 0.
    X_df = df.reindex(columns=feature_columns, fill_value=0.0)
    X = X_df.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float).values
    mean = np.array(scaler["mean"], dtype=np.float32)
    std = np.array(scaler["std"], dtype=np.float32)
    return ((X - mean) / std).astype(np.float32)


def row_score(row: pd.Series) -> str:
    return f"{safe_int(row.get('Davidson_Final_Points'))}-{safe_int(row.get('Opponent_Final_Points'))}"


def derive_labels(row: pd.Series) -> Dict[str, int]:
    result = str(row.get("Result", "")).upper()
    dav = safe_int(row.get("Davidson_Final_Points"))
    opp = safe_int(row.get("Opponent_Final_Points"))
    diff = abs(dav - opp)
    overtime = str(row.get("Overtime", "No")).lower() == "yes"
    comeback = str(row.get("Comeback_Win", "No")).lower() == "yes"
    halftime_leader = str(row.get("Halftime_Leader", ""))
    fg = safe_float(row.get("Davidson_FG_Percentage"))
    threes_made = safe_int(row.get("Davidson_3PT_Made"))
    ft = safe_float(row.get("Davidson_FT_Percentage"))
    assists = safe_int(row.get("Davidson_Assists"))
    paint = safe_int(row.get("Davidson_Points_In_Paint"))
    points_off_to = safe_int(row.get("Davidson_Points_Off_Turnovers"))
    bench = safe_int(row.get("Davidson_Bench_Points"))
    rebounds = safe_int(row.get("Davidson_Rebounds"))
    steals = safe_int(row.get("Davidson_Steals"))
    blocks = safe_int(row.get("Davidson_Blocks"))
    top_points = safe_int(row.get("Top_Scorer_Davidson_Points"))

    if diff <= 5:
        opening = 7
    elif result == "W" and diff >= 12:
        opening = 5
    elif result == "L" and diff >= 15:
        opening = 2
    elif result == "L":
        opening = 4
    elif comeback:
        opening = 3
    else:
        opening = 0

    if top_points >= 30:
        player = 6
    elif top_points >= 25:
        player = 2
    elif top_points >= 20:
        player = 1
    elif top_points >= 15:
        player = 4
    else:
        player = 3

    if bench >= 24:
        support = 3
    elif assists >= 17:
        support = 2
    elif result == "W":
        support = 0
    else:
        support = 1

    if overtime:
        flow = 5
    elif comeback:
        flow = 6
    elif halftime_leader == "Davidson" and result == "W":
        flow = 7
    elif diff <= 5:
        flow = 4
    elif result == "L":
        flow = 1
    elif fg >= 50:
        flow = 0
    else:
        flow = 3

    stat_candidates = [
        (fg / 60.0, 0),
        (threes_made / 12.0, 1),
        (ft / 90.0, 2),
        (assists / 25.0, 3),
        (paint / 40.0, 4),
        (points_off_to / 20.0, 5),
        (bench / 30.0, 6),
        (rebounds / 45.0, 7),
        (steals / 12.0, 8),
        (blocks / 8.0, 9),
    ]
    team_stat = max(stat_candidates, key=lambda x: x[0])[1]

    if result == "W" and diff >= 10:
        closing = 5
    elif result == "W":
        closing = 3
    elif diff <= 5:
        closing = 0
    elif result == "L":
        closing = 6
    else:
        closing = 2

    return {"opening": opening, "player": player, "support": support, "flow": flow, "team_stat": team_stat, "closing": closing}


class NarrativePlanDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_columns: List[str], scaler: Dict[str, List[float]]):
        self.df = df.reset_index(drop=True)
        self.X = transform_features(self.df, feature_columns, scaler)
        labels = [derive_labels(row) for _, row in self.df.iterrows()]
        self.y = {key: torch.tensor([label[key] for label in labels], dtype=torch.long) for key in labels[0].keys()}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        return torch.tensor(self.X[idx], dtype=torch.float32), {key: values[idx] for key, values in self.y.items()}


def plan_collate(batch):
    xs, ys = zip(*batch)
    X = torch.stack(xs)
    y_batch = {key: torch.stack([y[key] for y in ys]) for key in ys[0].keys()}
    return X, y_batch


class NarrativePlannerNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 96, dropout: float = 0.15, sampling_temperature: float = 0.85):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.sampling_temperature = sampling_temperature
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleDict({key: nn.Linear(hidden_dim, size) for key, size in OUTPUT_SIZES.items()})

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.backbone(x)
        return {key: head(z) for key, head in self.heads.items()}

    @torch.no_grad()
    def predict_plan(self, x: torch.Tensor, sample: bool = True) -> Dict[str, int]:
        self.eval()
        logits = self.forward(x.unsqueeze(0))
        plan = {}
        for key, value in logits.items():
            if sample:
                probs = torch.softmax(value / max(self.sampling_temperature, 1e-6), dim=1)
                plan[key] = int(torch.multinomial(probs, 1).item())
            else:
                plan[key] = int(value.argmax(dim=1).item())
        return plan



def apply_rule_overrides(row: pd.Series, plan: Dict[str, int]) -> Dict[str, int]:
    """Keep the model output logical for obvious game situations."""
    plan = dict(plan)
    result = str(row.get("Result", "")).upper()
    dav = safe_int(row.get("Davidson_Final_Points"))
    opp = safe_int(row.get("Opponent_Final_Points"))
    diff = abs(dav - opp)
    overtime = str(row.get("Overtime", "No")).lower() == "yes"
    comeback = str(row.get("Comeback_Win", "No")).lower() == "yes"
    halftime_leader = str(row.get("Halftime_Leader", ""))

    if overtime:
        plan["flow"] = 5
        if diff <= 5:
            plan["opening"] = 7
    elif comeback:
        plan["flow"] = 6
        plan["opening"] = 3
    elif result == "L" and diff >= 15:
        plan["opening"] = random.choice([2, 4, 10])
        plan["flow"] = random.choice([1, 9])
        plan["closing"] = random.choice([2, 6, 7])
    elif result == "L" and diff <= 5:
        plan["opening"] = random.choice([1, 7, 8])
        plan["flow"] = 4
        plan["closing"] = random.choice([0, 2, 6])
    elif result == "L":
        plan["opening"] = random.choice([2, 4, 8, 10])
        plan["flow"] = random.choice([1, 3, 9])
        plan["closing"] = random.choice([2, 6, 7])
    elif result == "W" and diff >= 12:
        plan["opening"] = random.choice([0, 5, 9])
        plan["closing"] = random.choice([3, 5])
    elif result == "W" and diff <= 5:
        plan["opening"] = random.choice([1, 7, 8])
        plan["flow"] = 4

    if halftime_leader == "Davidson" and result == "W" and not overtime:
        plan["flow"] = 7

    if safe_int(row.get("Davidson_Bench_Points")) >= 28:
        plan["support"] = 3
        plan["team_stat"] = 6

    if safe_int(row.get("Davidson_Steals")) >= 10:
        plan["team_stat"] = 8
    if safe_int(row.get("Davidson_Blocks")) >= 6:
        plan["team_stat"] = 9

    return plan

def format_narrative(row: pd.Series, plan: Dict[str, int]) -> str:
    mapping = {
        "opponent": str(get_field(row, "Opponent", "the opponent")),
        "score": row_score(row),
        "player": str(get_field(row, "Top_Scorer_Davidson", "A Davidson player")),
        "points": safe_int(get_field(row, "Top_Scorer_Davidson_Points", 0)),
        "rebounds": safe_int(get_field(row, "Top_Scorer_Davidson_Rebounds", 0)),
        "assists": safe_int(get_field(row, "Top_Scorer_Davidson_Assists", 0)),
        "second_player": str(get_field(row, "Second_Key_Player_Davidson", "Another Wildcat")),
        "second_statline": str(get_field(row, "Second_Key_Player_Davidson_Statline", "steady minutes")),
        "fg_pct": safe_float(get_field(row, "Davidson_FG_Percentage", 0.0)),
        "threes_made": safe_int(get_field(row, "Davidson_3PT_Made", 0)),
        "threes_attempted": safe_int(get_field(row, "Davidson_3PT_Attempted", 0)),
        "ft_pct": safe_float(get_field(row, "Davidson_FT_Percentage", 0.0)),
        "team_assists": safe_int(get_field(row, "Davidson_Assists", 0)),
        "paint_points": safe_int(get_field(row, "Davidson_Points_In_Paint", 0)),
        "points_off_turnovers": safe_int(get_field(row, "Davidson_Points_Off_Turnovers", 0)),
        "bench_points": safe_int(get_field(row, "Davidson_Bench_Points", 0)),
        "team_rebounds": safe_int(get_field(row, "Davidson_Rebounds", 0)),
        "steals": safe_int(get_field(row, "Davidson_Steals", 0)),
        "blocks": safe_int(get_field(row, "Davidson_Blocks", 0)),
    }
    plan = apply_rule_overrides(row, plan)
    p1 = " ".join([
        OPENING_TEMPLATES[plan["opening"]].format(**mapping),
        PLAYER_TEMPLATES[plan["player"]].format(**mapping),
        SUPPORT_TEMPLATES[plan["support"]].format(**mapping),
        FLOW_TEMPLATES[plan["flow"]].format(**mapping),
    ])
    p2 = " ".join([
        TEAM_STAT_TEMPLATES[plan["team_stat"]].format(**mapping),
        CLOSING_TEMPLATES[plan["closing"]].format(**mapping),
    ])
    return p1 + "\n\n" + p2


def evaluate_model(model: NarrativePlannerNet, loader: DataLoader, device: str) -> Tuple[float, Dict[str, float]]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    correct = {key: 0 for key in OUTPUT_SIZES.keys()}
    total = 0
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = {key: value.to(device) for key, value in y.items()}
            logits = model(X)
            loss = sum(criterion(logits[key], y[key]) for key in logits.keys())
            total_loss += float(loss.item())
            total += X.size(0)
            for key in logits.keys():
                correct[key] += int((logits[key].argmax(dim=1) == y[key]).sum().item())
    metrics = {key: correct[key] / max(total, 1) for key in correct.keys()}
    metrics["avg"] = sum(metrics.values()) / len(correct)
    return total_loss / max(len(loader), 1), metrics


def train_model(model, train_loader, val_loader, epochs, lr, device, patience):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.to(device)
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X = X.to(device)
            y = {key: value.to(device) for key, value in y.items()}
            optimizer.zero_grad()
            logits = model(X)
            loss = sum(criterion(logits[key], y[key]) for key in logits.keys())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            train_loss += float(loss.item())
        train_loss /= max(len(train_loader), 1)
        val_loss, metrics = evaluate_model(model, val_loader, device)
        print(f"Epoch {epoch:03d} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | avg planning acc {metrics['avg']:.2f}")
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping: validation loss stopped improving.")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val_loss


def save_checkpoint(path, model, feature_columns, scaler, hidden_dim, dropout, sampling_temperature):
    torch.save({
        "model_state_dict": model.state_dict(),
        "feature_columns": feature_columns,
        "scaler": scaler,
        "config": {"hidden_dim": hidden_dim, "dropout": dropout, "sampling_temperature": sampling_temperature},
    }, path)


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    feature_columns = checkpoint["feature_columns"]
    scaler = checkpoint["scaler"]
    config = checkpoint.get("config", {})
    model = NarrativePlannerNet(
        len(feature_columns),
        config.get("hidden_dim", 96),
        config.get("dropout", 0.15),
        config.get("sampling_temperature", 0.85),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, feature_columns, scaler


def run_train(args):
    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)
    feature_columns = get_numeric_feature_columns(train_df)
    scaler = fit_scaler(train_df, feature_columns)
    train_dataset = NarrativePlanDataset(train_df, feature_columns, scaler)
    val_dataset = NarrativePlanDataset(val_df, feature_columns, scaler)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=plan_collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=plan_collate)
    model = NarrativePlannerNet(len(feature_columns), args.hidden_dim, args.dropout, args.sampling_temperature)
    best_val_loss = train_model(model, train_loader, val_loader, args.epochs, args.learning_rate, args.device, args.patience)
    save_checkpoint(args.model_path, model, feature_columns, scaler, args.hidden_dim, args.dropout, args.sampling_temperature)
    print(f"\nTraining complete. Best validation loss: {best_val_loss:.4f}")
    print(f"Saved model to {args.model_path}")


def generate_for_row(model, feature_columns, scaler, row, device, sample=True):
    X = transform_features(pd.DataFrame([row]), feature_columns, scaler)
    x = torch.tensor(X[0], dtype=torch.float32).to(device)
    plan = model.predict_plan(x, sample=sample)
    return format_narrative(row, plan)


def run_generate(args):
    model, feature_columns, scaler = load_checkpoint(args.model_path, args.device)
    df = pd.read_csv(args.input_csv)
    row_index = random.randint(0, len(df) - 1) if str(args.row).lower() == "random" else int(args.row)
    row = df.iloc[row_index]
    narrative = generate_for_row(model, feature_columns, scaler, row, args.device, sample=not args.deterministic)
    print(f"\nSelected row: {row_index}")
    print(f"Opponent: {get_field(row, 'Opponent', 'Unknown')}")
    print(f"Score: {row_score(row)}")
    if "Generated_Narrative_Target" in df.columns:
        print("\n=== TARGET NARRATIVE FROM DATASET ===\n")
        print(row["Generated_Narrative_Target"])
    print("\n=== AI GENERATED NARRATIVE ===\n")
    print(narrative)


def run_generate_file(args):
    model, feature_columns, scaler = load_checkpoint(args.model_path, args.device)
    df = pd.read_csv(args.input_csv)
    df["AI_Generated_Narrative"] = [
        generate_for_row(model, feature_columns, scaler, row, args.device, sample=not args.deterministic)
        for _, row in df.iterrows()
    ]
    df.to_csv(args.output_csv, index=False)
    print(f"Generated narratives saved to {args.output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Small-data PyTorch AI for Davidson basketball narratives.")
    parser.add_argument("--mode", choices=["train", "generate", "generate_file"], default="train")
    parser.add_argument("--train_csv", type=str, default="train_dataset.csv")
    parser.add_argument("--val_csv", type=str, default="validation_dataset.csv")
    parser.add_argument("--input_csv", type=str, default="validation_dataset.csv")
    parser.add_argument("--output_csv", type=str, default="ai_generated_output.csv")
    parser.add_argument("--model_path", type=str, default="best_narrative_planner.pt")
    parser.add_argument("--row", type=str, default="random")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--sampling_temperature", type=float, default=0.85)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--deterministic", action="store_true", help="Use argmax instead of sampling for repeatable output.")
    args = parser.parse_args()
    if args.mode == "train":
        run_train(args)
    elif args.mode == "generate":
        run_generate(args)
    elif args.mode == "generate_file":
        run_generate_file(args)


if __name__ == "__main__":
    main()
