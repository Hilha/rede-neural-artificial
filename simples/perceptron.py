import numpy as np

#
# Definições
#
entradas = np.array([[0,0], [0,1], [1,0], [1,1]]) # entradas operador lógico AND
saidas = np.array([0,0,0,1]) # saídas operador lógico AND

#entradas = np.array([[0,0],[0,1],[1,0],[1,1]]) # entradas operador lógico OR
#saidas = np.array([0,1,1,1]) # saídas operador lógico OR

#entradas = np.array([[0,0],[0,1],[1,0],[1,1]]) # entradas operador lógico XOR
#saidas = np.array([0,1,1,0]) # saídas operador lógico XOR

pesos = np.array([0.0, 0.0])
taxaAprendizagem = 0.1

# Função de Ativação (step function)
def stepFunction(soma):
    if(soma >= 1):
        return 1
    return 0

# recebe um registro (entradas),
# realiza o produto escalar, multiplica as entradas pelos pesos e irá fazer a soma, utilizando dot (numpy)
# aplica stepfunction retornando o valor da saída
def calculaSaida(registro):
    _soma = registro.dot(pesos)
    return stepFunction(_soma)

# treina a rede
# faz toda a iteração nos registros, entradas saídas e pesos
# também realiza o processo de ajuste de pesos (encontra o melhor conjunto de pesos)
def treinar():
    _erroTotal = 1
    while(_erroTotal != 0): # para quando o erro for igual a zero
        _erroTotal = 0
        for saida in range(len(saidas)):
            _saidaCalculada = calculaSaida(np.asarray(entradas[saida])) # calcula saída com os pesos atuais
            _erro = abs(saidas[saida] - _saidaCalculada) # compara a saída esperada com a saída calculada e soma ao erro
            _erroTotal += _erro

            # atualização nos pesos
            for peso in range(len(pesos)):
                # atualiza os pesos de acordo com: peso(n+1) = pesoAtual + (taxaAprendizagem * entrada * erro)
                pesos[peso] = pesos[peso] + (taxaAprendizagem * entradas[saida][peso] * _erro)
                print('peso atualizado: ' + str(pesos[peso]))

        print('total de erros: ' + str(_erroTotal))

treinar()
print('Rede Neural treinada!')
print(calculaSaida(entradas[0]))
print(calculaSaida(entradas[1]))
print(calculaSaida(entradas[2]))
print(calculaSaida(entradas[3]))