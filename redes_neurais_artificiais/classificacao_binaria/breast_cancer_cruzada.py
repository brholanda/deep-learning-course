import pandas as pdimport kerasfrom keras.models import Sequentialfrom keras.layers import Dense, Dropoutfrom keras.wrappers.scikit_learn import KerasClassifierfrom sklearn.model_selection import cross_val_score# Leitura das basesprevisores = pd.read_csv('entradas_breast.csv')classe = pd.read_csv('saidas_breast.csv')def criar_rede():    classificador = Sequential()    classificador.add(Dense(units = 16, activation = 'selu',                             kernel_initializer = 'random_uniform', input_dim = 30))    classificador.add(Dropout(0.2))    classificador.add(Dense(units = 16, activation = 'selu',                             kernel_initializer = 'random_uniform'))    classificador.add(Dropout(0.2))    classificador.add(Dense(units = 16, activation = 'selu',                             kernel_initializer = 'random_uniform'))    classificador.add(Dropout(0.2))    classificador.add(Dense(units = 16, activation = 'selu',                             kernel_initializer = 'random_uniform'))    classificador.add(Dropout(0.2))    classificador.add(Dense(units = 16, activation = 'selu',                             kernel_initializer = 'random_uniform'))    classificador.add(Dropout(0.2))    classificador.add(Dense(units = 1, activation = 'sigmoid'))    #otimizador = keras.optimizers.Adam(lr = 0.001, decay = 0.0001, clipvalue = 0.5)    otimizador = keras.optimizers.Nadam(learning_rate = 0.001, beta_2 = 0.9999, clipvalue = 0.5)    classificador.compile(optimizer = otimizador, loss = 'binary_crossentropy',                          metrics = ['binary_accuracy'])        return classificadorclassificador = KerasClassifier(build_fn = criar_rede,                                epochs = 200,                                batch_size = 30)resultados = cross_val_score(estimator = classificador,                             X = previsores, y = classe,                             cv = 10, scoring = 'accuracy')media = resultados.mean()desvio = resultados.std()