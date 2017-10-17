#Implementação de uma RNA Perceptron de 1 (uma) camada

#
# Função Somatória
#
def soma(entradas, pesos):
    _soma = 0
    for i in range(3):
        #print(entradas[i])
        #print(pesos[i])
        _soma += entradas[i] * pesos[i]
    return _soma

#
# Função de Ativação
#
def stepFunction(soma):
    if(soma >= 1):
        return 1 # neurônio ativado
    return 0     # neurônio não ativado

# ======================================== || Testes
entradas = [1, 7, 5]
pesos = [0.8, 0.1, 0]

soma = soma(entradas, pesos)

resultado = stepFunction(soma)
print(resultado)