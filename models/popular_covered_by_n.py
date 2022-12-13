from __future__ import annotations
from itertools import islice, cycle
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import mode
from scipy.sparse import csr_matrix

from rectools.dataset import Dataset
from rectools import Columns


class PopularCoveredByNPercent:
    def __init__(self, percent: float, max_k: int = 10) -> None:
        self.percent = percent
        if self.percent <= 0 or self.percent > 1:
            raise ValueError("'percent' must be float in [0; 1].")
        self.max_k = max_k

    def _get_top_items_covered_users(self, matrix: csr_matrix, n_users: int) -> List:
        item_set = []
        covered_users = np.zeros(matrix.shape[0], dtype=bool) 
        while covered_users.sum() < n_users: 
            top_item = mode(matrix[~covered_users].indices)[0][0] 
            item_set.append(top_item)
            covered_users += np.maximum.reduceat(
                matrix.indices==top_item, 
                matrix.indptr[:-1], 
                dtype=bool
            ) 
        return item_set

    def fit(self, dataset: Dataset) -> PopularCoveredByNPercent:
        matrix = dataset.get_user_item_matrix()
        n_users = int(self.percent * dataset.interactions.df[Columns.User].nunique())
        item_set = self._get_top_items_covered_users(matrix, n_users)
        self.recommendations = dataset.item_id_map.convert_to_external(item_set)[:self.max_k]
        return self

    def recommend(self, users: pd.Series) -> List:
        return list(islice(cycle([self.recommendations]), len(users)))
        

if __name__ == "__main__":
    interactions = pd.read_csv('data/interactions_processed.csv',  parse_dates=['last_watch_dt'])
    interactions.rename(
        columns={
            'last_watch_dt': Columns.Datetime,
            'total_dur': Columns.Weight
        },
        inplace=True
    )
    items = pd.read_csv('data/items_processed.csv')
    items = items[items[Columns.Item].isin(interactions[Columns.Item])]
    users = pd.read_csv('data/users_processed.csv')
    print(f"Interactions shape: {interactions.shape}")
    print(f"Items shape: {items.shape}")
    print(f"Users shape: {users.shape}")

    dataset = Dataset.construct(interactions_df=interactions)
    pop_model = PopularCoveredByNPercent(
        percent=0.7,
        max_k=10
    )
    pop_model.fit(dataset)

    recs = pd.DataFrame({Columns.User: interactions[Columns.User].unique()})
    recs[Columns.Item] = pop_model.recommend(recs[Columns.User])
    recs = recs.explode(Columns.Item)
    recs[Columns.Rank] = recs.groupby(Columns.User).cumcount() + 1
    recs = pd.merge(
        recs,  
        items[['item_id', 'title']], 
        on='item_id',
        how='left'
    )
    print(recs.head(10))
