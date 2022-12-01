import pandas as pd
import numpy as np

from models import PopularModel
from time_range_splitter import TimeRangeSplitter


def compute_metrics(train, test, recs) -> pd.Series:
    top_N = 10
    result = {}
    test_recs = test.set_index(['user_id', 'item_id']).join(recs.set_index(['user_id', 'item_id']))
    test_recs = test_recs.sort_values(by=['user_id', 'rank'])

    test_recs['users_item_count'] = test_recs.groupby(level='user_id')['rank'].transform(np.size)
    test_recs['reciprocal_rank'] = (1 / test_recs['rank']).fillna(0)
    test_recs['cumulative_rank'] = test_recs.groupby(level='user_id').cumcount() + 1
    test_recs['cumulative_rank'] = test_recs['cumulative_rank'] / test_recs['rank']
    
    users_count = test_recs.index.get_level_values('user_id').nunique()
    result[f'MAP@{top_N}'] = (test_recs['cumulative_rank'] / test_recs['users_item_count']).sum() / users_count    
    return pd.Series(result)


if __name__ == '__main__':
    # read dataframes
    interactions = pd.read_csv('data/interactions_processed.csv',  parse_dates=['last_watch_dt'])
    items = pd.read_csv('data/items_processed.csv')
    users = pd.read_csv('data/users_processed.csv')

    # cross-validation
    n_folds = 3
    n_units = 7
    unit = "D"
    validation_results = pd.DataFrame()
    last_date = interactions['last_watch_dt'].max().normalize()
    start_date = last_date - pd.Timedelta(days=n_folds*n_units)

    cv = TimeRangeSplitter(
        start_date=start_date,
        periods=n_folds+1,
        freq='W',
        filter_already_seen=True,
        filter_cold_items=True,
        filter_cold_users=True,
    )
    fold_iterator = cv.split(
        df = interactions,
        user_column='user_id',
        item_column='item_id',
        datetime_column='last_watch_dt',
        fold_stats=True
    )
    for i_fold, (train_ids, test_ids, fold_info) in enumerate(fold_iterator):
        print(f"\n==================== Fold {i_fold+1}")
        df_train = interactions.iloc[train_ids]
        df_test = interactions.iloc[test_ids]
        print(f'Train: {df_train.shape[0]} | Test:{df_test.shape[0]}')
        model = PopularModel(days=10, dttm_column='last_watch_dt', item_column='item_id')
        model.fit(df_train)
        recs = pd.DataFrame({'user_id': df_test['user_id'].unique()})
        recs['item_id'] = model.recommend(recs['user_id'])
        recs = recs.explode('item_id')
        recs['rank'] = recs.groupby('user_id').cumcount() + 1

        fold_result = compute_metrics(df_train, df_test, recs)

        validation_results = validation_results.append(fold_result, ignore_index=True)

    print(validation_results)
    # selection of hyperparameters

    # plot graphics





    



