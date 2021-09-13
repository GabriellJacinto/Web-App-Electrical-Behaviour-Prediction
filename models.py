import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from joblib import dump

df = pd.read_csv('clean_data.csv', index_col='Unnamed: 0')

df_NOR = df.copy()
df_NOR.iint = (df_NOR.iint-df_NOR.iint.min()) / (df_NOR.iint.max()-df_NOR.iint.min())
df_NOR.tplh = (df_NOR.tplh-df_NOR.tplh.min())/(df_NOR.tplh.max()-df_NOR.tplh.min())
df_NOR.tphl = (df_NOR.tphl-df_NOR.tphl.min())/(df_NOR.tphl.max()-df_NOR.tphl.min())

X = df_NOR.drop(columns=['tphl', 'tplh', 'iint'])
y = df_NOR[['tphl', 'tplh', 'iint']]

scaler = StandardScaler().fit(X)
X_sc = scaler.transform(X)
seed = 42

rf_iint = RandomForestRegressor(n_estimators=150, max_depth=15, random_state=seed)
rf_iint.fit(X_sc, y.iint)

rf_tphl = RandomForestRegressor(n_estimators=25, max_depth=10,random_state=seed)
rf_tphl.fit(X_sc, y.tphl)

rf_tplh = RandomForestRegressor(n_estimators=150,max_depth=10,random_state=seed)
rf_tplh.fit(X_sc, y.tplh)


dump(rf_iint, 'iint.gz', compress=3)
dump(rf_tphl, 'tphl.gz', compress=3)
dump(rf_tplh, 'tplh.gz', compress=3)