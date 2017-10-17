#Implementação de uma RNA Perceptron de 1 (uma) camada

# Importações
import numpy as np

#
# Função Somatória
#
def soma(entradas, pesos):
    return entradas.dot(pesos) # dot product / produto escalar

#
# Função de Ativação
#
def stepFunction(soma):
    if(soma >= 1):
        return 1 # neurônio ativado
    return 0     # neurônio não ativado

# ======================================== || Testes
entradas = np.array([1, 7, 5])
pesos = np.array([0.8, 0.1, 0])

soma = soma(entradas, pesos)

resultado = stepFunction(soma)
print(resultado)