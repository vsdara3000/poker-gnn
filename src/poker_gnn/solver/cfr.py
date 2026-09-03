"""Tabular CFR — ground-truth solver for Kuhn / Leduc."""


class TabularCFR:
    def iterate(self, game, iterations: int):
        raise NotImplementedError

    def average_strategy(self):
        raise NotImplementedError
