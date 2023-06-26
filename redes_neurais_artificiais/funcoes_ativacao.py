import numpy as np

def stepFunction(soma):
    if (soma >= 1):
        return 1
    return 0

stepValue = stepFunction(-1)

def sigmoidFunction(soma):
    return 1 / (1 + np.exp(-soma))

sigmoidValue = sigmoidFunction(2.1)

def tangentFunction(soma):
    return (np.exp(soma) - np.exp(-soma)) / (np.exp(soma) + np.exp(-soma))

tangentValue = tangentFunction(2.1)

def reluFunction(soma):
    if (soma >= 0):
        return soma
    return 0

reluValue = reluFunction(2.1)

def linearFunction(soma):
    return soma

linearValue = linearFunction(2.1)

def softmaxFunction(vetor):
    exponencial = np.exp(vetor)
    print(exponencial)
    print(exponencial)
    print(exponencial.sum())
    print(148.4131591 / exponencial.sum())
    return exponencial / exponencial.sum()

valores = [5.0, 2.0, 1.3]
softmaxValue = softmaxFunction(valores)

teste = 5 * 0.2 + 2 * 0.5 + 1 * 0.1
