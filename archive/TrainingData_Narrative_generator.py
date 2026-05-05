"""
Davidson Automated Narrative Generator
-------------------------------------

Inputs:
1. Basketball analytics CSV
2. Sentence bank CSV

Output:
A new CSV with Generated_Narrative_Target filled in.

Expected files:
- Basketball_Training_Dataset.csv
- davidson_sentence_bank_v2.csv

Run:
python davidson_narrative_generator.py
"""

import ast
import operator
import random
import re
from pathlib import Path

import pandas as pd


ANALYTICS_CSV = "Basketball_Training_Dataset(1).csv"
SENTENCE_BANK_CSV = "sentence_bank.csv"
OUTPUT_CSV = "Full_Training_data_FIXED.csv"

DAVIDSON_NAME = "Davidson"


# -----------------------------
# Helper functions
# -----------------------------

def clean_value(value):
    """Return None for missing/blank values; otherwise return value."""
    if pd.isna(value):
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


def safe_num(value, default=0):
    """Convert a value to float safely."""
    value = clean_value(value)
    if value is None:
        return default
    if isinstance(value, str):
        value = value.replace("%", "").strip()
        if value.lower() in {"yes", "true"}:
            return 1
        if value.lower() in {"no", "false"}:
            return 0
    try:
        return float(value)
    except Exception:
        return default


def safe_int(value, default=0):
    return int(round(safe_num(value, default)))


def yes_true(value):
    """Interpret yes/true/1 values."""
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"yes", "true", "1", "y"}
    return bool(value)


def score_string(row):
    """Return Davidson score first, opponent score second."""
    return f"{safe_int(row.get('Davidson_Final_Points'))}-{safe_int(row.get('Opponent_Final_Points'))}"


def second_half_score(row):
    return f"{safe_int(row.get('Davidson_Second_Half_Points'))}-{safe_int(row.get('Opponent_Second_Half_Points'))}"


def classify_outcome(row):
    result = str(row.get("Result", "")).strip().upper()
    diff = safe_num(row.get("Point_Differential"))

    if result == "W":
        if diff >= 15:
            return "Win_Blowout"
        if 1 <= diff <= 8:
            return "Win_Close"
        return "Win_Controlled"

    if result == "L":
        if abs(diff) <= 8:
            return "Loss_Close"
        if abs(diff) >= 15:
            return "Loss_Blowout"
        return "Loss_Competitive"

    return "Any"


def classify_game_flow(row):
    result = str(row.get("Result", "")).strip().upper()
    diff = safe_num(row.get("Point_Differential"))
    halftime_leader = str(row.get("Halftime_Leader", "")).strip()
    second_half_diff = safe_num(row.get("Second_Half_Differential"))
    lead_changes = safe_num(row.get("Lead_Changes"))
    times_tied = safe_num(row.get("Times_Tied"))

    if yes_true(row.get("Comeback_Win")) or (result == "W" and halftime_leader == "Opponent"):
        return "Comeback"
    if yes_true(row.get("Overtime")) or safe_num(row.get("Overtime_Periods")) > 0:
        return "Overtime"
    if result == "W" and halftime_leader == "Davidson" and safe_num(row.get("Halftime_Margin")) >= 10:
        return "Early_Dominance"
    if lead_changes >= 6 or times_tied >= 5:
        return "Back_and_Forth"
    if abs(diff) <= 8:
        return "Late_Composure"
    if second_half_diff > 0:
        return "Second_Half_Response"
    return "Any"


def classify_player_type(row):
    pts = safe_num(row.get("Top_Scorer_Davidson_Points"))
    reb = safe_num(row.get("Top_Scorer_Davidson_Rebounds"))
    ast = safe_num(row.get("Top_Scorer_Davidson_Assists"))

    if safe_num(row.get("Double_Doubles_Davidson")) >= 1:
        return "Double_Double"
    if pts >= 25:
        return "Elite_Performance"
    if pts >= 20:
        return "Star_Performance"
    if pts >= 10 and reb >= 5 and ast >= 3:
        return "Well_Rounded"
    if pts < 20 and safe_num(row.get("Davidson_Assists")) >= 15:
        return "Balanced_Contribution"
    return "Standard_Leader"


def classify_team_type(row):
    if safe_num(row.get("Davidson_Bench_Points")) >= 20:
        return "Bench_Strength"
    if safe_num(row.get("Davidson_3PT_Percentage")) >= 38:
        return "Strong_Three_Point_Shooting"
    if safe_num(row.get("Davidson_FG_Percentage")) >= 48:
        return "Strong_Shooting"
    if safe_num(row.get("Rebound_Differential")) >= 5:
        return "Rebound_Dominance"
    if safe_num(row.get("Turnover_Differential")) <= -3:
        return "Turnover_Control"
    if safe_num(row.get("Opponent_Final_Points")) <= 65:
        return "Defensive_Strength"
    if safe_num(row.get("Davidson_Fast_Break_Points")) >= 10:
        return "Transition_Offense"
    if safe_num(row.get("Davidson_Second_Chance_Points")) >= 10:
        return "Second_Chance_Offense"
    if safe_num(row.get("Davidson_Points_In_Paint")) >= 34:
        return "Paint_Scoring"
    return "Any"


def narrative_strategy(row):
    result = str(row.get("Result", "")).strip().upper()
    return "Win_Positive" if result == "W" else "Loss_Positive"


# -----------------------------
# Trigger matching
# -----------------------------

def condition_matches(trigger, row):
    """
    Lightweight evaluator for the trigger language used in the sentence bank.

    It handles the common trigger formats we created:
    - Result == W
    - Point_Differential >= 15
    - ABS(Point_Differential) <= 8
    - BETWEEN a AND b
    - IS NOT NULL
    - AND / OR combinations
    - Overtime == Yes
    """
    if not isinstance(trigger, str) or trigger.strip() == "" or trigger.strip().lower() == "any":
        return True

    parts_or = re.split(r"\s+OR\s+", trigger)
    return any(_and_conditions_match(part, row) for part in parts_or)


def _and_conditions_match(trigger_part, row):
    conditions = re.split(r"\s+AND\s+", trigger_part)
    return all(_single_condition_match(cond.strip(), row) for cond in conditions if cond.strip())


def _single_condition_match(cond, row):
    # Assembly-rule triggers should not be used as normal sentence conditions
    if "Use one Opening" in cond or "avoid harsh negative wording" in cond:
        return False

    # IS NOT NULL
    m = re.match(r"(.+?)\s+IS NOT NULL", cond)
    if m:
        field = m.group(1).strip()
        return clean_value(row.get(field)) is not None

    # BETWEEN
    m = re.match(r"(.+?)\s+BETWEEN\s+(-?\d+\.?\d*)\s+AND\s+(-?\d+\.?\d*)", cond)
    if m:
        field = m.group(1).strip()
        low = float(m.group(2))
        high = float(m.group(3))
        value = safe_num(row.get(field))
        return low <= value <= high

    # ABS(field) comparison
    m = re.match(r"ABS\((.+?)\)\s*(<=|>=|==|<|>)\s*(-?\d+\.?\d*)", cond)
    if m:
        field = m.group(1).strip()
        op = m.group(2)
        target = float(m.group(3))
        value = abs(safe_num(row.get(field)))
        return compare(value, op, target)

    # Standard field comparison
    m = re.match(r"(.+?)\s*(==|<=|>=|<|>)\s*(.+)", cond)
    if m:
        field = m.group(1).strip()
        op = m.group(2)
        raw_target = m.group(3).strip()

        actual = row.get(field)

        # String comparisons
        if raw_target in {"W", "L", "Yes", "No", "Davidson", "Opponent"}:
            return str(actual).strip().upper() == raw_target.upper()

        # Boolean-ish comparisons
        if raw_target in {"1", "0"} and str(actual).strip().lower() in {"true", "false", "yes", "no", "1", "0"}:
            return safe_num(actual) == float(raw_target)

        # Numeric comparisons
        return compare(safe_num(actual), op, float(raw_target))

    return False


def compare(a, op, b):
    ops = {
        "==": operator.eq,
        "<=": operator.le,
        ">=": operator.ge,
        "<": operator.lt,
        ">": operator.gt,
    }
    return ops[op](a, b)


# -----------------------------
# Template filling
# -----------------------------

def build_placeholder_values(row, role=None):
    """Build placeholder values.

    Important: [REBOUNDS] and [ASSISTS] appear in both player and team
    templates. Therefore their meaning has to depend on the sentence role.
    For Body_Player, they refer to the top scorer's player line.
    For Body_Team_Stat, they refer to Davidson's team totals.
    """
    opponent = row.get("Opponent", "the opponent")

    player_values = {
        "REBOUNDS": safe_int(row.get("Top_Scorer_Davidson_Rebounds")),
        "ASSISTS": safe_int(row.get("Top_Scorer_Davidson_Assists")),
    }
    team_values = {
        "REBOUNDS": safe_int(row.get("Davidson_Rebounds")),
        "ASSISTS": safe_int(row.get("Davidson_Assists")),
    }

    # Default to player values unless we are filling a team-stat sentence.
    ambiguous_values = team_values if role == "Body_Team_Stat" else player_values

    values = {
        "DAVIDSON": DAVIDSON_NAME,
        "OPPONENT": opponent,
        "SCORE": score_string(row),
        "SECOND_HALF_SCORE": second_half_score(row),
        "RECORD": row.get("Davidson_Record_After", ""),
        "PLAYER": row.get("Top_Scorer_Davidson", "A Davidson standout"),
        "POINTS": safe_int(row.get("Top_Scorer_Davidson_Points")),
        "REBOUNDS": ambiguous_values["REBOUNDS"],
        "ASSISTS": ambiguous_values["ASSISTS"],
        "PLAYER_REBOUNDS": safe_int(row.get("Top_Scorer_Davidson_Rebounds")),
        "PLAYER_ASSISTS": safe_int(row.get("Top_Scorer_Davidson_Assists")),
        "TEAM_REBOUNDS": safe_int(row.get("Davidson_Rebounds")),
        "TEAM_ASSISTS": safe_int(row.get("Davidson_Assists")),
        "SECOND_PLAYER": row.get("Second_Key_Player_Davidson", "Another Davidson contributor"),
        "SECOND_STATLINE": row.get("Second_Key_Player_Davidson_Statline", "key contributions"),
        "THIRD_PLAYER": row.get("Third_Key_Player_Davidson", "A third Davidson contributor"),
        "THIRD_STATLINE": row.get("Third_Key_Player_Davidson_Statline", "important minutes"),
        "FG_PERCENTAGE": safe_num(row.get("Davidson_FG_Percentage")),
        "OPP_FG": safe_num(row.get("Opponent_FG_Percentage")),
        "THREES_MADE": safe_int(row.get("Davidson_3PT_Made")),
        "THREES_ATTEMPTED": safe_int(row.get("Davidson_3PT_Attempted")),
        "FT_PERCENTAGE": safe_num(row.get("Davidson_FT_Percentage")),
        "POINTS_IN_PAINT": safe_int(row.get("Davidson_Points_In_Paint")),
        "FAST_BREAK_POINTS": safe_int(row.get("Davidson_Fast_Break_Points")),
        "SECOND_CHANCE_POINTS": safe_int(row.get("Davidson_Second_Chance_Points")),
        "REBOUND_DIFFERENTIAL": safe_int(row.get("Rebound_Differential")),
        "STEALS": safe_int(row.get("Davidson_Steals")),
        "BLOCKS": safe_int(row.get("Davidson_Blocks")),
        "OPPONENT_POINTS": safe_int(row.get("Opponent_Final_Points")),
        "POINTS_OFF_TURNOVERS": safe_int(row.get("Davidson_Points_Off_Turnovers")),
        "HALFTIME_MARGIN": safe_int(row.get("Halftime_Margin")),
        "BENCH_POINTS": safe_int(row.get("Davidson_Bench_Points")),
        "KEY_RUN": row.get("Key_Run_1", "key scoring run"),
    }

    # Formatting cleanup for percentages
    for key in ["FG_PERCENTAGE", "OPP_FG", "FT_PERCENTAGE"]:
        if isinstance(values[key], float):
            values[key] = f"{values[key]:.1f}".rstrip("0").rstrip(".")

    return values


def fill_template(template, row, role=None):
    values = build_placeholder_values(row, role)

    def replace_match(match):
        key = match.group(1)
        return str(values.get(key, match.group(0)))

    return re.sub(r"\[([A-Z0-9_]+)\]", replace_match, template)


# -----------------------------
# Sentence selection
# -----------------------------

def filter_candidates(sentence_bank, row, role):
    outcome = classify_outcome(row)
    flow = classify_game_flow(row)
    player_type = classify_player_type(row)
    team_type = classify_team_type(row)
    strategy = narrative_strategy(row)

    candidates = sentence_bank[sentence_bank["Role"] == role].copy()

    if candidates.empty:
        return candidates

    def row_matches_template(template_row):
        # Category matching: exact or Any or broad Win/Loss.
        outcome_type = str(template_row.get("Outcome_Type", "Any"))
        flow_type = str(template_row.get("Game_Flow_Type", "Any"))
        player = str(template_row.get("Player_Type", "Any"))
        team = str(template_row.get("Team_Type", "Any"))
        strat = str(template_row.get("Narrative_Strategy", "Davidson_Positive"))

        result = str(row.get("Result", "")).strip().upper()
        broad_outcome_ok = (
            outcome_type == "Any"
            or outcome_type == outcome
            or (outcome_type == "Win" and result == "W")
            or (outcome_type == "Loss" and result == "L")
        )

        return (
            broad_outcome_ok
            and (flow_type == "Any" or flow_type == flow)
            and (player == "Any" or player == player_type)
            and (team == "Any" or team == team_type)
            and (strat in {"Davidson_Positive", strategy})
            and condition_matches(str(template_row.get("Trigger_Logic", "Any")), row)
        )

    mask = candidates.apply(row_matches_template, axis=1)
    candidates = candidates[mask].copy()

    if not candidates.empty:
        candidates["Priority"] = candidates["Priority"].fillna(3).astype(float)
        candidates = candidates.sort_values(["Priority", "Sentence_ID"])
    return candidates


def choose_sentence(sentence_bank, row, role, used_ids=None):
    used_ids = used_ids or set()
    candidates = filter_candidates(sentence_bank, row, role)

    if candidates.empty:
        return None, None

    candidates = candidates[~candidates["Sentence_ID"].isin(used_ids)]
    if candidates.empty:
        return None, None

    # Pick among top-priority candidates for some variation.
    best_priority = candidates["Priority"].min()
    top = candidates[candidates["Priority"] == best_priority]
    selected = top.sample(1, random_state=random.randint(0, 999999)).iloc[0]

    sentence = fill_template(selected["Template"], row, role)
    return sentence, selected["Sentence_ID"]


def generate_narrative(row, sentence_bank):
    used = set()

    opening, sid = choose_sentence(sentence_bank, row, "Opening", used)
    if sid: used.add(sid)

    player_sentence, sid = choose_sentence(sentence_bank, row, "Body_Player", used)
    if sid: used.add(sid)

    flow_sentence, sid = choose_sentence(sentence_bank, row, "Body_Flow", used)
    if sid: used.add(sid)

    stat_sentence, sid = choose_sentence(sentence_bank, row, "Body_Team_Stat", used)
    if sid: used.add(sid)

    closing, sid = choose_sentence(sentence_bank, row, "Closing", used)
    if sid: used.add(sid)

    # Fallbacks
    if not opening:
        opponent = row.get("Opponent", "the opponent")
        if str(row.get("Result", "")).upper() == "W":
            opening = f"Davidson secured a strong result against {opponent}, winning {score_string(row)}."
        else:
            opening = f"Davidson competed hard against {opponent}, falling {score_string(row)}."

    if not player_sentence:
        player_sentence = f"{row.get('Top_Scorer_Davidson', 'A Davidson standout')} led the way with {safe_int(row.get('Top_Scorer_Davidson_Points'))} points."

    if not stat_sentence:
        stat_sentence = "Davidson found positive stretches through effort, execution and continued competitiveness."

    if not closing:
        closing = "Davidson will look to build on the positives from this performance."

    # Two-paragraph format
    paragraph_1_sentences = [opening, player_sentence]
    if flow_sentence:
        paragraph_1_sentences.append(flow_sentence)

    paragraph_2_sentences = [stat_sentence, closing]

    paragraph_1 = " ".join(paragraph_1_sentences)
    paragraph_2 = " ".join(paragraph_2_sentences)

    return paragraph_1 + "\n\n" + paragraph_2


def main():
    analytics_path = Path(ANALYTICS_CSV)
    sentence_bank_path = Path(SENTENCE_BANK_CSV)

    if not analytics_path.exists():
        raise FileNotFoundError(f"Could not find {analytics_path}. Put it in the same folder as this script.")
    if not sentence_bank_path.exists():
        raise FileNotFoundError(f"Could not find {sentence_bank_path}. Put it in the same folder as this script.")

    analytics = pd.read_csv(analytics_path)
    sentence_bank = pd.read_csv(sentence_bank_path)

    # Remove assembly rule rows from normal generation
    sentence_bank = sentence_bank[sentence_bank["Role"] != "Assembly_Rule"].copy()

    random.seed(42)

    analytics["Generated_Narrative_Target"] = analytics.apply(
        lambda row: generate_narrative(row, sentence_bank),
        axis=1
    )

    analytics.to_csv(OUTPUT_CSV, index=False)
    print(f"Generated narratives for {len(analytics)} games.")
    print(f"Saved output to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
