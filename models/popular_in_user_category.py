from __future__ import annotations
from typing import List

import pandas as pd

from rectools import Columns


class PopularInUserCategory:

    def __init__(self, cat_feature: str, max_k: int = 10) -> None:
        self.cat_feature = cat_feature
        self.max_k = max_k
        self.nan_value = f"nan_{self.cat_feature}"

    def _get_top_by_cat(self, counts_by_cat: pd.DataFrame):
        top_cat_items = []
        for cat in counts_by_cat[self.cat_feature].unique():
            top_items = (
                counts_by_cat[counts_by_cat[self.cat_feature] == cat]
                .sort_values(0, ascending=False)
                .head(self.max_k)
                .item_id
                .values
            )
            top_cat_items.append([cat, top_items])
        top_cat_items = pd.DataFrame(
            top_cat_items, columns=[self.cat_feature, Columns.Item]
        )
        return top_cat_items

    def fit(self, df_train: pd.DataFrame) -> PopularInUserCategory:
        train = df_train.fillna(value={self.cat_feature: self.nan_value})
        counts_by_cat = (
            train
            .groupby([self.cat_feature, Columns.Item])
            .size()
            .to_frame()
            .reset_index()
        )
        self.top_by_cat_items = self._get_top_by_cat(counts_by_cat)
        return self

    def recommend(self, df_users: pd.DataFrame) -> List:
        try:
            df_users[self.cat_feature]
        except KeyError:
            print(f"'users' must contain '{self.cat_feature}' feature!")
            return
        users = df_users.fillna(value={self.cat_feature: self.nan_value})
        recs = pd.merge(
            users,
            self.top_by_cat_items,
            on=self.cat_feature,
            how='left'
        )
        recs.drop(columns=[self.cat_feature], inplace=True)
        return recs


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
    CAT_FEATURE = "age"
    train = pd.merge(
        interactions,
        users[[Columns.User, CAT_FEATURE]], 
        on=Columns.User,
        how='left'
    ) 
    pop_model = PopularInUserCategory(
        cat_feature='age',
        max_k=10
    )
    pop_model.fit(train)

    recs = pd.DataFrame({Columns.User: interactions[Columns.User].unique()})
    recs = pd.merge(recs, users[[Columns.User, CAT_FEATURE]], on=Columns.User, how='left')
    recs = pop_model.recommend(recs)
    recs = recs.explode(Columns.Item)
    recs[Columns.Rank] = recs.groupby(Columns.User).cumcount() + 1
    recs = pd.merge(
        recs,  
        items[['item_id', 'title']], 
        on='item_id',
        how='left'
    )
    print(recs.head(10))