import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler

pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

df = pd.read_csv('clientes-v2-tratados.csv')

print(df.head())

# tira as informacoes que não utiliza
# df = df.drop(['data', 'estado', 'nivel_educacao', 'numero_filho', 'estado_civil', 'area_atuacao'], axis=1)

# deixando idade e salario pra ficar mais facil a analise
df = df[['idade', 'salario']]

# Normalização - MinMaxScaler
scaler = MinMaxScaler()
df['idadeMinMaxScaler'] = scaler.fit_transform(df[['idade']])
df['salarioMinMaxScaler'] = scaler.fit_transform(df[['salario']])

min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
df['idadeMinMaxScaler_mm'] = scaler.fit_transform(df[['idade']])
df['salarioMinMaxScaler_mm'] = scaler.fit_transform(df[['salario']])

# Padronização - StandardScaler
scaler = StandardScaler()
df['idadeStandardScaler'] = scaler.fit_transform(df[['idade']])
df['salarioStandardScaler'] = scaler.fit_transform(df[['salario']])

# Padronização - RobustScaler
scaler = RobustScaler()
df['idadeRobustScaler'] = scaler.fit_transform(df[['idade']])
df['salarioRobustScaler'] = scaler.fit_transform(df[['salario']])

print(df.head(15))

print("MinMaxScaler (De 0 a 1):")
print("Idade - Min: {:.4f} Max: {:.4f} Mean: {:.4f} Std: {:.4f}".format(df['idadeMinMaxScaler'].min(),df['idadeMinMaxScaler'].max(),df['idadeMinMaxScaler'].mean(), df['idadeMinMaxScaler'].std()))
print("Salario - Min: {:.4f} Max: {:.4f} Mean: {:.4f} Std: {:.4f}".format(df['salarioMinMaxScaler'].min(),df['salarioMinMaxScaler'].max(), df['salarioMinMaxScaler'].mean(), df['idadeMinMaxScaler'].std()))

print("\nMinMaxScaler (De -1 a 1):")
print("Idade - Min: {:.4f} Max: {:.4f} Mean: {:.4f} Std: {:.4f}".format(df['idadeMinMaxScaler_mm'].min(),df['idadeMinMaxScaler_mm'].max(),df['idadeMinMaxScaler_mm'].mean(), df['idadeMinMaxScaler_mm'].std()))
print("Salario - Min: {:.4f} Max: {:.4f} Mean: {:.4f} Std: {:.4f}".format(df['salarioMinMaxScaler_mm'].min(),df['salarioMinMaxScaler_mm'].max(), df['salarioMinMaxScaler_mm'].mean(), df['salarioMinMaxScaler_mm'].std()))

print("\nStandardScaler (Ajuste a media a 0 e desvio padrão 1):")
print("Idade - Min: {:.4f}, Max: {:.4f}, Mean: {:.18f}, Std: {:.4f}".format(df['idadeStandardScaler'].min(), df['idadeStandardScaler'].max(), df['idadeStandardScaler'].mean(), df['idadeStandardScaler'].std()))
print("Salario - Min: {:.4f}, Max: {:.4f}, Mean: {:.18f}, Std: {:.4f}".format(df['salarioStandardScaler'].min(), df['salarioStandardScaler'].max(), df['salarioStandardScaler'].mean(), df['salarioStandardScaler'].std()))

print("\nRobustScaler (De 0 a 1):")
print("Idade - Min: {:.4f}, Max: {:.4f}, Mean: {:.4f}, Std: {:.4f}".format(df['idadeRobustScaler'].min(), df['idadeRobustScaler'].max(), df['idadeRobustScaler'].mean(), df['idadeRobustScaler'].std()))
print("Idade - Min: {:.4f}, Max: {:.4f}, Mean: {:.4f}, Std: {:.4f}".format(df['salarioRobustScaler'].min(), df['salarioRobustScaler'].max(), df['salarioRobustScaler'].mean(), df['salarioRobustScaler'].std()))
