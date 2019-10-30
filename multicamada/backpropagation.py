"""
    File: backpropagation
    Created by user Matheus Hilha
    2019/25/10 - 04:35:22.0
"""

import math
import random
import numpy

def criar_linha():
    print("-" * 80)

def rand(a, b):
    return (b - a) * random.random() + a

# função de ativação sigmoide
def funcao_ativacao_sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

# função de ativação sigmoide
def derivada_funcao_ativacao_sigmoid(x):
    return x * (1 - x)

# função de ativação tangente hiperbolica
def funcao_ativacao_tang_hip(x):
    return math.tanh(x)

# derivada da tangente hiperbolica
def derivada_funcao_ativacao_tang_hip(x):
    t = funcao_ativacao_tang_hip(x)
    return 1 - t ** 2

class RedeNeural:
    def __init__(self, neuronios_entrada, neuronios_ocultos, neuronios_saida):
        # camada de entrada
        self.neuronios_entrada = neuronios_entrada + 1  # +1 por causa do no do bias
        # camada oculta
        self.neuronios_ocultos = neuronios_ocultos
        # camada de saía
        self.neuronios_saida = neuronios_saida
        # quantidade máxima de iterações
        self.max_iteracoes = 1000
        # taxa de aprendizado
        self.taxa_aprendizado = 0.5
        # momento
        self.momentum = 0.1

        # ativa as camadas de neurônios
        # cria uma matriz de uma linha pela quantidade de neurônios (1xN)
        self.ativacao_entrada = numpy.ones(self.neuronios_entrada) # ones enche as matrizes de 1
        self.ativacao_ocultos = numpy.ones(self.neuronios_ocultos)
        self.ativacao_saida = numpy.ones(self.neuronios_saida)

        # contém os resultados das ativações de saída
        self.resultados_ativacao_saida = numpy.ones(self.neuronios_saida)

        # criar a matriz de pesos, preenchidas com zeros
        self.pesos_entradas = numpy.zeros((self.neuronios_entrada, self.neuronios_ocultos))
        self.pesos_ocultos = numpy.zeros((self.neuronios_ocultos, self.neuronios_saida))

        # adicionar os valores dos pesos
        # vetor de pesos da camada de entrada - intermediaria
        for _entrada in range(self.neuronios_entrada):
            for _oculto in range(self.neuronios_ocultos):
                self.pesos_entradas[_entrada][_oculto] = rand(-0.2, 0.2)

        # vetor de pesos da camada intermediaria - saida
        for _oculto in range(self.neuronios_ocultos):
            for _saida in range(self.neuronios_saida):
                self.pesos_ocultos[_oculto][_saida] = rand(-2.0, 2.0)

        # última mudança nos pesos para o momento
        self.camadas_entradas = numpy.zeros((self.neuronios_entrada, self.neuronios_ocultos))
        self.camadas_ocultas = numpy.zeros((self.neuronios_ocultos, self.neuronios_saida))

    def fase_forward(self, entradas):
        # ativa as entradas: -1 por causa do bias
        for i in range(self.neuronios_entrada - 1):
            self.ativacao_entrada[i] = entradas[i]

        # calcula as ativações dos neurônios da camada escondida
        for j in range(self.neuronios_ocultos):
            soma = 0
            for i in range(self.neuronios_entrada):
                soma = soma + self.ativacao_entrada[i] * self.pesos_entradas[i][j]
            self.ativacao_ocultos[j] = funcao_ativacao_tang_hip(soma)
            #self.ativacao_ocultos[j] = funcao_ativacao_sigmoid(soma)

        # calcula as ativações dos neurônios da camada de saída
        # Note que as saidas dos neurônios da camada oculta fazem o papel de entrada
        # para os neurôios da camada de saída.
        for j in range(self.neuronios_saida):
            soma = 0
            for i in range(self.neuronios_ocultos):
                soma = soma + self.ativacao_ocultos[i] * self.pesos_ocultos[i][j]
            self.ativacao_saida[j] = funcao_ativacao_tang_hip(soma)
            #self.ativacao_saida[j] = funcao_ativacao_sigmoid(soma)

        return self.ativacao_saida

    def fase_backward(self, saidas_desejadas):
        # calcular os gradientes locais dos neurônios da camada de saida
        output_deltas = numpy.zeros(self.neuronios_saida)
        erro = 0
        for i in range(self.neuronios_saida):
            erro = saidas_desejadas[i] - self.ativacao_saida[i]
            output_deltas[i] = derivada_funcao_ativacao_tang_hip(self.ativacao_saida[i]) * erro
            #output_deltas[i] = derivada_funcao_ativacao_sigmoid(self.ativacao_saida[i]) * erro

        # calcular os gradientes locais dos neurônios da camada oculta
        hidden_deltas = numpy.zeros(self.neuronios_ocultos)
        for i in range(self.neuronios_ocultos):
            erro = 0
            for j in range(self.neuronios_saida):
                erro = erro + output_deltas[j] * self.pesos_ocultos[i][j]
            hidden_deltas[i] = derivada_funcao_ativacao_tang_hip(self.ativacao_ocultos[i]) * erro
            #hidden_deltas[i] = derivada_funcao_ativacao_sigmoid(self.ativacao_ocultos[i]) * erro

        # a partir da ultima camada até a camada de entrada
        # os neurônios da camada atual ajustam seus pesos e reduzem os seus erros
        for i in range(self.neuronios_ocultos):
            for j in range(self.neuronios_saida):
                change = output_deltas[j] * self.ativacao_ocultos[i]
                self.pesos_ocultos[i][j] = self.pesos_ocultos[i][j] + (self.taxa_aprendizado * change) + (self.momentum * self.camadas_ocultas[i][j])
                self.camadas_ocultas[i][j] = change

        # atualizar os pesos da primeira camada
        for i in range(self.neuronios_entrada):
            for j in range(self.neuronios_ocultos):
                change = hidden_deltas[j] * self.ativacao_entrada[i]
                self.pesos_entradas[i][j] = self.pesos_entradas[i][j] + (self.taxa_aprendizado * change) + (self.momentum * self.camadas_entradas[i][j])
                self.camadas_entradas[i][j] = change

        # calcula o erro
        erro = 0
        for i in range(len(saidas_desejadas)):
            erro = erro + 0.5 * (saidas_desejadas[i] - self.ativacao_saida[i]) ** 2
        return erro

    def test(self, entradas_saidas):
        for p in entradas_saidas:
            array = self.fase_forward(p[0])
            print("Entradas: " + str(p[0]) + ' - Saída encontrada/fase forward: ' + str(array[0]))

    def treinar(self, entradas_saidas):
        for i in range(self.max_iteracoes):
            erro = 0
            for p in entradas_saidas:
                entradas = p[0]
                saidas_desejadas = p[1]
                self.fase_forward(entradas)
                erro = erro + self.fase_backward(saidas_desejadas)
            if i % 100 == 0:
                print("Erro = %2.3f" % erro)

def iniciar():
    # Ensinar a rede a reconhecer o padrão XOR
    entradas_saidas = [
        [[0, 0], [0]],
        [[0, 1], [1]],
        [[1, 0], [1]],
        [[1, 1], [0]]
    ]

    # cria rede neural com duas entradas, duas ocultas e um neurônio de saida
    rede = RedeNeural(2, 2, 1)
    criar_linha()
    # treinar com os padrões
    rede.treinar(entradas_saidas)
    # testar
    criar_linha()
    rede.test(entradas_saidas)


if __name__ == '__main__':
    iniciar()
