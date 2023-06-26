import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

base = pd.read_csv('autos.csv', encoding='ISO-8859-1')
base = base.drop('dateCrawled', axis=1)
base = base.drop('dateCreated', axis=1)
base = base.drop('nrOfPictures', axis=1)
base = base.drop('postalCode', axis=1)
base = base.drop('lastSeen', axis=1)

base['name'].value_counts()
base = base.drop('name', axis=1)
base['seller'].value_counts()
base = base.drop('seller', axis=1)
base['offerType'].value_counts()
base = base.drop('offerType', axis=1)

#identifica dados com valor de preço muito baixo ou muito alto
inconsistentes1 = base.loc[base.price <= 10]
base = base.loc[base.price > 10]
inconsistentes2 = base.loc[base.price > 350000]
base = base.loc[base.price < 350000]

#identifica valores null
base.loc[pd.isnull(base['vehicleType'])]
base['vehicleType'].value_counts() #limousine
base.loc[pd.isnull(base['gearbox'])]
base['gearbox'].value_counts() #manuell
base.loc[pd.isnull(base['model'])]
base['model'].value_counts() #golf
base.loc[pd.isnull(base['fuelType'])]
base['fuelType'].value_counts() #benzin
base.loc[pd.isnull(base['notRepairedDamage'])]
base['notRepairedDamage'].value_counts() #nein

#preenche valores null com os respectivos valores mais comuns
valores = {'vehicleType': 'limousine',
           'gearbox': 'manuell', 
           'model': 'golf', 
           'fuelType': 'benzin', 
           'notRepairedDamage': 'nein'}
base = base.fillna(value = valores)

previsores = base.iloc[:, 1:13].values
preco_real = base.iloc[:, 0].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

#altera valores string para numericos
labelencoder_previsores = LabelEncoder()
previsores[:, 0] = labelencoder_previsores.fit_transform(previsores[:, 0])
previsores[:, 1] = labelencoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 3] = labelencoder_previsores.fit_transform(previsores[:, 3])
previsores[:, 5] = labelencoder_previsores.fit_transform(previsores[:, 5])
previsores[:, 8] = labelencoder_previsores.fit_transform(previsores[:, 8])
previsores[:, 9] = labelencoder_previsores.fit_transform(previsores[:, 9])
previsores[:, 10] = labelencoder_previsores.fit_transform(previsores[:, 10])

#inclui uma coluna para cada possivel valor das colunas indicadas
# 0 0 0 0
# 2 0 1 0
# 3 0 0 1
columntransformer = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0,1,3,5,8,9,10])],
    remainder='passthrough'
)
previsores = columntransformer.fit_transform(previsores)

regressor = Sequential()
regressor.add(Dense(units=158, activation='relu', input_dim=316))
regressor.add(Dense(units=158, activation='relu'))
regressor.add(Dense(1, activation='linear'))
regressor.compile(loss='mean_absolute_error', optimizer='adam',
                  metrics=['mean_absolute_error'])
regressor.fit(previsores, preco_real, batch_size=300, epochs=100)

previsoes = regressor.predict(previsores)
preco_real.mean()
previsores.mean()