import pandas as pd
from keras.layers import Dense, Dropout, Activation, Input
from keras.models import Model

#importação da base e remoção de colunas irrelevantes
base = pd.read_csv('games.csv')
base = base.drop('Other_Sales', axis=1)
base = base.drop('NA_Sales', axis=1)
base = base.drop('EU_Sales', axis=1)
base = base.drop('JP_Sales', axis=1)
base = base.drop('Developer', axis=1)

#remoção de valores inconsistentes
base = base.dropna(axis=0)
base = base.loc[base['Global_Sales'] > 1]

nome_jogos = base.Name
base = base.drop('Name', axis=1)

#previsores e resultados esperados
previsores = base.iloc[:, [0,1,2,3,5,6,7,8,9]].values
venda = base.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

#transformação de colunas string em numerico
labelencoder = LabelEncoder()
previsores[:, 0] = labelencoder.fit_transform(previsores[:, 0])
previsores[:, 2] = labelencoder.fit_transform(previsores[:, 2])
previsores[:, 3] = labelencoder.fit_transform(previsores[:, 3])
previsores[:, 8] = labelencoder.fit_transform(previsores[:, 8])

#transformação de novos valores numericos em colunas
columntransformer = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0,2,3,8])],
    remainder='passthrough'
)
previsores = columntransformer.fit_transform(previsores).toarray()

#criação da rede
camada_entrada = Input(shape=(99,))
optimizer = Activation(activation='sigmoid')
camada_oculta1 = Dense(units = 50, activation = optimizer)(camada_entrada)
camada_oculta2 = Dense(units = 50, activation = optimizer)(camada_oculta1)
optimizer_saida = Activation(activation='linear')
camada_saida = Dense(units = 1, activation = optimizer_saida)(camada_oculta2)

regressor = Model(inputs = camada_entrada,
                  outputs = camada_saida)

#treinamento
regressor.compile(optimizer = 'adam',
                  loss = 'mse')
regressor.fit(previsores, venda,
              epochs = 5000, batch_size = 100)

#resultado
previsao = regressor.predict(previsores)
