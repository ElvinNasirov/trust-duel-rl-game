"""
Core game engine for Trust Duel: Person vs Agent.

The game is based on the Iterated Prisoner's Dilemma.
"""

from __future__ import annotations

import copy
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


Action = str
ACTIONS: Tuple[Action, Action] = ("C", "D")

PAYOFFS: Dict[Tuple[Action, Action], Tuple[int, int]] = {
    ("C", "C"): (3, 3),
    ("C", "D"): (0, 5),
    ("D", "C"): (5, 0),
    ("D", "D"): (1, 1),
}


def flip_action(action: Action) -> Action:
    """Flip C to D and D to C."""
    if action == "C":
        return "D"
    if action == "D":
        return "C"
    raise ValueError(f"Unknown action: {action}")


def get_payoff(action_a: Action, action_b: Action) -> Tuple[int, int]:
    """Return rewards for Player A and Player B."""
    if (action_a, action_b) not in PAYOFFS:
        raise ValueError(f"Invalid action pair: {(action_a, action_b)}")
    return PAYOFFS[(action_a, action_b)]


def simulate_match(
    strategy_a,
    strategy_b,
    rounds: int = 200,
    noise: float = 0.0,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Simulate a repeated Prisoner's Dilemma match.

    Parameters
    ----------
    strategy_a, strategy_b:
        Objects with:
        - name attribute
        - choose_action(my_history, opponent_history) method

    rounds:
        Number of repeated rounds.

    noise:
        Probability that a selected action is accidentally flipped.
        Example: C may become D because of misunderstanding/noise.

    seed:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Round-by-round game history.
    """
    if not 0 <= noise <= 1:
        raise ValueError("noise must be between 0 and 1")

    rng = np.random.default_rng(seed)

    if hasattr(strategy_a, "reset"):
        strategy_a.reset()
    if hasattr(strategy_b, "reset"):
        strategy_b.reset()

    history_a: List[Action] = []
    history_b: List[Action] = []
    rows = []

    total_a = 0
    total_b = 0

    for round_idx in range(1, rounds + 1):
        intended_a = strategy_a.choose_action(history_a, history_b)
        intended_b = strategy_b.choose_action(history_b, history_a)

        action_a = flip_action(intended_a) if rng.random() < noise else intended_a
        action_b = flip_action(intended_b) if rng.random() < noise else intended_b

        reward_a, reward_b = get_payoff(action_a, action_b)

        total_a += reward_a
        total_b += reward_b

        history_a.append(action_a)
        history_b.append(action_b)

        rows.append(
            {
                "round": round_idx,
                "player_a": strategy_a.name,
                "player_b": strategy_b.name,
                "intended_a": intended_a,
                "intended_b": intended_b,
                "action_a": action_a,
                "action_b": action_b,
                "reward_a": reward_a,
                "reward_b": reward_b,
                "cumulative_a": total_a,
                "cumulative_b": total_b,
            }
        )

    return pd.DataFrame(rows)


def tournament(
    strategies: Iterable,
    rounds: int = 200,
    noise: float = 0.0,
    include_self: bool = True,
    seed: int | None = 42,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Run a round-robin tournament among strategies.

    Each strategy plays each other strategy once.
    Optionally, each strategy also plays against a copy of itself.
    """
    strategies = list(strategies)
    scores = {strategy.name: 0 for strategy in strategies}
    matches: Dict[str, pd.DataFrame] = {}

    rng = np.random.default_rng(seed)

    for i, strategy_a in enumerate(strategies):
        start_j = i if include_self else i + 1

        for j in range(start_j, len(strategies)):
            strategy_b = strategies[j]

            if i == j and not include_self:
                continue

            # Deepcopy prevents strategies with random/internal state from leaking across matches.
            a = copy.deepcopy(strategy_a)
            b = copy.deepcopy(strategy_b)

            match_seed = int(rng.integers(0, 1_000_000))
            match_df = simulate_match(a, b, rounds=rounds, noise=noise, seed=match_seed)

            match_name = f"{a.name} vs {b.name}"
            matches[match_name] = match_df

            scores[a.name] += int(match_df["reward_a"].sum())
            scores[b.name] += int(match_df["reward_b"].sum())

    ranking = (
        pd.DataFrame(
            [{"strategy": name, "total_score": score} for name, score in scores.items()]
        )
        .sort_values("total_score", ascending=False)
        .reset_index(drop=True)
    )

    ranking["rank"] = ranking.index + 1
    ranking["average_score_per_match_round"] = ranking["total_score"] / (
        rounds * len(strategies)
    )

    return ranking[["rank", "strategy", "total_score", "average_score_per_match_round"]], matches
