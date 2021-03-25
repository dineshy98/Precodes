#data

data["DayOfMonth"] = data['placement date'].apply(lambda x: x.day)
data["Month"] = data['placement date'].apply(lambda x: x.month)
data["WeekDay"] = data['placement date'].apply(lambda x: x.weekday())     
data["DayOfYear"] = data['placement date'].apply(lambda x: x.timetuple().tm_yday)
data["DayCount"] = data['placement date'].apply(lambda x: x.toordinal())

#### 2.Categorical Groupby - on Continous

data['total_price'] = data['total_price']*data['no of items']

for col in cat_cols:
    col = [col]
    col_name = str(col[0]) + "_"
    all_df = data[["price"]+ col]
    gdf = all_df.groupby(col)["price"].agg(["mean", "std", "max"]).reset_index()
    gdf.columns = col + [col_name+"_price_mean", col_name+"_price_std", col_name+"_price_max"]
    data = pd.merge(data, gdf, on=col, how="left")
    
    
####Multi-Categorical Groupby
for col in [
            'placement date',
            'departure city',
            
            [ "segment","sub-class"],
            ["sub-class", 'departure city'],
            ['sub-class', 'placement date'],
    
    
            ["sub-class", "checkin_date_year"],
            ["memberid", "checkin_date_month"],
            ["resort_id", "checkin_date_year"],
            ["resort_id", "checkin_date_month"],
            ["resort_id", "checkin_date_year", "checkin_date_month"],
    
            ["resort_id", "state_code_residence", "checkin_date"],
            ["resort_id", "state_code_residence", "checkout_date"],
    
            ["resort_id", "checkin_date_year", "checkin_date_week"],
            ["resort_id", "state_code_residence", "checkin_date_year", "checkin_date_week"],
            ["resort_id", "state_code_residence", "checkin_date_year", "checkin_date_month"],
    
    
            ["booking_date", "checkin_date"],
            ["booking_date", "checkin_date", "resort_id"],
    

           ]:
    if not isinstance(col, list):
        col = [col]
    col_name = "_".join(col)
    all_df = pd.concat([train_df[["reservation_id"]+ col], test_df[["reservation_id"]+ col]])
    gdf = all_df.groupby(col)["reservation_id"].count().reset_index()
    gdf.columns = col + [col_name+"_count"]
    train_df = pd.merge(train_df, gdf, on=col, how="left")
    
    
    
    
from catboost import CatBoostRegressor
model=CatBoostRegressor(iterations=10000, depth=3, learning_rate=0.1, loss_function='RMSE')
model.fit(X_tr1, y_tr1,eval_set=(X_tst1, y_tst1),plot=True)
print(r2_score(y_tst1,model.predict(X_tst1)))
print(np.sqrt(mean_squared_error(y_tst1,model.predict(X_tst1))))

import matplotlib.pyplot as plt
import lightgbm as lgb 

fig, ax = plt.subplots(figsize=(12,30))
lgb.plot_importance(lgbm, max_num_features=100, height=0.8, ax=ax)
ax.grid(False)
plt.title("LightGBM - Feature Importance", fontsize=15)
plt.show()

