import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from joblib import dump

df = pd.read_csv('clean_data.csv', index_col='Unnamed: 0')

X = df.drop(columns=['tphl', 'tplh', 'iint'])
y = df[['tphl', 'tplh', 'iint']]

scaler_NOR = MinMaxScaler().fit(y)
y_sc = scaler_NOR.transform(y)
y_sc = pd.DataFrame(y_sc)

scaler_SC = StandardScaler().fit(X)
X_sc = scaler_SC.transform(X)


seed = 42

print("[IINT] Starting...")
rf_iint = DecisionTreeRegressor(max_depth=15, random_state=seed)
rf_iint.fit(X_sc, y_sc[2])
print("[IINT] Done")

print("[TPHL] Starting...")
rf_tphl = DecisionTreeRegressor(max_depth=10,random_state=seed)
rf_tphl.fit(X_sc, y_sc[0])
print("[TPHL] Done")

print("[TPLH] Starting...")
rf_tplh = DecisionTreeRegressor(max_depth=10,random_state=seed)
rf_tplh.fit(X_sc, y_sc[1])
print("[TPLH] Done")

dump(scaler_NOR, 'scaler_NOR.gz', compress=3)
dump(scaler_SC, 'scaler_SC.gz', compress=3)
dump(rf_iint, 'iint.gz', compress=5)
dump(rf_tphl, 'tphl.gz', compress=3)
dump(rf_tplh, 'tplh.gz', compress=3)