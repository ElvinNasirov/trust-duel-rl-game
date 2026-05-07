"""
Fixed strategies for Trust Duel.
"""

from __future__ import annotations

import numpy as np


class BaseStrategy:
    """Base class for all fixed strategies."""

    name = "Base Strategy"

    def reset(self) -> None:
        """Reset internal state if a strategy uses it."""
        return None

    def choose_action(self, my_history, opponent_history) -> str:
        raise NotImplementedError


class AlwaysCooperate(BaseStrategy):
    name = "Always Cooperate"

    def choose_action(self, my_history, opponent_history) -> str:
        return "C"


class AlwaysDefect(BaseStrategy):
    name = "Always Defect"

    def choose_action(self, my_history, opponent_history) -> str:
        return "D"


class RandomStrategy(BaseStrategy):
    name = "Random"

    def __init__(self, p_defect: float = 0.5, seed: int | None = None):
        if not 0 <= p_defect <= 1:
            raise ValueError("p_defect must be between 0 and 1")
        self.p_defect = p_defect
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def choose_action(self, my_history, opponent_history) -> str:
        return "D" if self.rng.random() < self.p_defect else "C"


class TitForTat(BaseStrategy):
    name = "Tit for Tat"

    def choose_action(self, my_history, opponent_history) -> str:
        if len(opponent_history) == 0:
            return "C"
        return opponent_history[-1]


class GrimTrigger(BaseStrategy):
    """
    Friedman / Grim Trigger strategy.

    Starts with cooperation.
    If opponent defects once, defects forever.
    """

    name = "Grim Trigger"

    def choose_action(self, my_history, opponent_history) -> str:
        if "D" in opponent_history:
            return "D"
        return "C"


class TitForTwoTats(BaseStrategy):
    """
    Starts with cooperation.
    Defects only if opponent defected in the previous two rounds.
    """

    name = "Tit for Two Tats"

    def choose_action(self, my_history, opponent_history) -> str:
        if len(opponent_history) < 2:
            return "C"
        if opponent_history[-1] == "D" and opponent_history[-2] == "D":
            return "D"
        return "C"


class ForgivingTitForTat(BaseStrategy):
    """
    Tit for Tat with forgiveness.

    If opponent defected in the previous round, this strategy usually defects too,
    but sometimes forgives and cooperates.
    """

    name = "Forgiving Tit for Tat"

    def __init__(self, forgiveness: float = 0.1, seed: int | None = None):
        if not 0 <= forgiveness <= 1:
            raise ValueError("forgiveness must be between 0 and 1")
        self.forgiveness = forgiveness
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def choose_action(self, my_history, opponent_history) -> str:
        if len(opponent_history) == 0:
            return "C"

        if opponent_history[-1] == "D":
            return "C" if self.rng.random() < self.forgiveness else "D"

        return "C"


class SuspiciousTitForTat(BaseStrategy):
    """
    Starts with defection, then copies opponent's previous move.
    Useful as a contrast with regular Tit for Tat.
    """

    name = "Suspicious Tit for Tat"

    def choose_action(self, my_history, opponent_history) -> str:
        if len(opponent_history) == 0:
            return "D"
        return opponent_history[-1]
