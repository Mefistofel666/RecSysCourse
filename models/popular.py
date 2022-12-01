from __future__ import annotations
from itertools import islice, cycle

import pandas as pd
import numpy as np

# TODO: add docstring
# TODO: inheritance from sklearn.basemodel or smth else
class PopularModel:

    def __init__(
        self, 
        item_column: str,
        dttm_column: str,
        days: int = 30,
        max_k: int = 10
    ) -> None:
        self.item_column = item_column
        self.dttm_column = dttm_column
        self.days = days
        self.max_k = max_k

    def fit(self, df: pd.DataFrame) -> PopularModel:
        if not self.days:
            self.days = (
                df[self.dt_column].max().normalize() - df[self.dt_column].min().normalize()
                ).days

        min_date = df[self.dttm_column].max().normalize() - pd.DateOffset(days=self.days)
        self.recommendations = (
            df
            .loc[df[self.dttm_column] > min_date, self.item_column]
            .value_counts()
            .head(self.max_k)
            .index
            .values
        )
        return self

    def recommend(self, users = None) -> np.ndarray:
        if users is None:
            return self.recommendations
        
        return list(islice(cycle([self.recommendations]), len(users)))

