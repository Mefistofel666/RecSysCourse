from __future__ import annotations
from itertools import islice, cycle
from typing import List

import pandas as pd

from rectools.dataset import Dataset
from rectools import Columns


class PopularModel:
    def __init__(self, days: int = 30, max_k: int = 10) -> None:
        self.days = days
        self.max_k = max_k

    def fit(self, dataset: Dataset) -> PopularModel:
        interactions_df = dataset.interactions.df
        max_ts = interactions_df[Columns.Datetime].max().normalize()
        min_ts = interactions_df[Columns.Datetime].min().normalize()
        if not self.days:
            self.days = (max_ts - min_ts).days

        min_date = max_ts - pd.DateOffset(days=self.days)
        self.recommendations = (
            interactions_df
            .loc[interactions_df[Columns.Datetime] > min_date, Columns.Item]
            .value_counts()
            .head(self.max_k)
            .index
            .values
        )
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
    pop_model = PopularModel(
        days=7,
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
