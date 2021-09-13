import pandas as pd

df = pd.read_csv('treated_data.csv', index_col='Unnamed: 0')
df = df[(2e-12<=df.tphl)&(df.tphl<=2.9e-11)] #filtrando para deixar no intervalo acima
df = df[(1e-12<=df.tplh)&(df.tplh<=6.8e-11)]
df = df[(-3e-15<=df.iint)]

df.to_csv('clean_data.csv')