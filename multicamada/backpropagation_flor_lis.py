"""
    File: backpropagation
    Created by user Joshua Raiser
    2017/17/10 - 04:35:22.0
"""
import math
import random
import numpy

from bases import bases

from sklearn import datasets

def criar_linha():
    print("-" * 110)

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
    def __init__(self, neuronios_entrada, neuronios_ocultos, neuronios_saida, iteracoes, aprendizagem, momento):
        # camada de entrada
        self.neuronios_entrada = neuronios_entrada + 1  # +1 por causa do no do bias
        # camada oculta
        self.neuronios_ocultos = neuronios_ocultos
        # camada de saía
        self.neuronios_saida = neuronios_saida
        # quantidade máxima de iterações
        self.max_iteracoes = iteracoes
        # taxa de aprendizado
        self.taxa_aprendizado = aprendizagem
        # momento
        self.momentum = momento

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
        # ativa as entradas, -1 por causa do bias
        for i in range(self.neuronios_entrada - 1):
            self.ativacao_entrada[i] = entradas[i]

        # calcula as ativações dos neurônios da camada escondida
        for j in range(self.neuronios_ocultos):
            soma = 0
            for i in range(self.neuronios_entrada):
                soma = soma + self.ativacao_entrada[i] * self.pesos_entradas[i][j]
            #self.ativacao_ocultos[j] = funcao_ativacao_tang_hip(soma)
            self.ativacao_ocultos[j] = funcao_ativacao_sigmoid(soma)

        # calcula as ativações dos neurônios da camada de saída
        # Note que as saidas dos neurônios da camada oculta fazem o papel de entrada
        # para os neurôios da camada de saída.
        for j in range(self.neuronios_saida):
            soma = 0
            for i in range(self.neuronios_ocultos):
                soma = soma + self.ativacao_ocultos[i] * self.pesos_ocultos[i][j]
            #self.ativacao_saida[j] = funcao_ativacao_tang_hip(soma)
            self.ativacao_saida[j] = funcao_ativacao_sigmoid(soma)

        return self.ativacao_saida

    def fase_backward(self, saidas_desejadas):
        # calcular os gradientes locais dos neurônios da camada de saida
        output_deltas = numpy.zeros(self.neuronios_saida)
        erro = 0
        for i in range(self.neuronios_saida):
            erro = saidas_desejadas[i] - self.ativacao_saida[i]
            #output_deltas[i] = derivada_funcao_ativacao_tang_hip(self.ativacao_saida[i]) * erro
            output_deltas[i] = derivada_funcao_ativacao_sigmoid(self.ativacao_saida[i]) * erro

        # calcular os gradientes locais dos neurônios da camada oculta
        hidden_deltas = numpy.zeros(self.neuronios_ocultos)
        for i in range(self.neuronios_ocultos):
            erro = 0
            for j in range(self.neuronios_saida):
                erro = erro + output_deltas[j] * self.pesos_ocultos[i][j]
            #hidden_deltas[i] = derivada_funcao_ativacao_tang_hip(self.ativacao_ocultos[i]) * erro
            hidden_deltas[i] = derivada_funcao_ativacao_sigmoid(self.ativacao_ocultos[i]) * erro

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
            erro = erro + self.taxa_aprendizado * (saidas_desejadas[i] - self.ativacao_saida[i]) ** 2
        return erro

    def test(self, entradas_saidas):
        for p in entradas_saidas:
            array = self.fase_forward(p[0])
            print("Entradas: " + str(p[0]) + ' - Saída encontrada/fase forward: ' + str(array))

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

    # 125 dados para aprendizagem (4 entradas/ 3 saídas)
    flor_lis = [
        [[4.8, 3.1, 1.6, 0.2], [1, 0, 0]],
        [[5.5, 2.4, 3.8, 1.1], [ 0, 1, 0]],
        [[7.4, 2.8, 6.1, 1.9], [ 0, 0, 1]],
        [[5.4, 3.4, 1.5, 0.4], [ 1, 0, 0]],
        [[5.5, 2.4, 3.7, 1.0], [ 0, 1, 0]],
        [[7.9, 3.8, 6.4, 2.0], [ 0, 0, 1]],
        [[5.2, 4.1, 1.5, 0.1], [ 1, 0, 0]],
        [[5.8, 2.7, 3.9, 1.2], [ 0, 1, 0]],
        [[6.4, 2.8, 5.6, 2.2], [ 0, 0, 1]],
        [[5.5, 4.2, 1.4, 0.2], [ 1, 0, 0]],
        [[6.0, 2.7, 5.1, 1.6], [ 0, 1, 0]],
        [[6.3, 2.8, 5.1, 1.5], [ 0, 0, 1]],
        [[4.9, 3.1, 1.5, 0.2], [ 1, 0, 0]],
        [[5.4, 3.0, 4.5, 1.5], [ 0, 1, 0]],
        [[6.1, 2.6, 5.6, 1.4], [ 0, 0, 1]],
        [[5.0, 3.2, 1.2, 0.2], [ 1, 0, 0]],
        [[6.0, 3.4, 4.5, 1.6], [ 0, 1, 0]],
        [[7.7, 3.0, 6.1, 2.3], [ 0, 0, 1]],
        [[5.5, 3.5, 1.3, 0.2], [ 1, 0, 0]],
        [[6.7, 3.1, 4.7, 1.5], [ 0, 1, 0]],
        [[6.3, 3.4, 5.6, 2.4], [ 0, 0, 1]],
        [[4.9, 3.6, 1.4, 0.1], [ 1, 0, 0]],
        [[6.3, 2.3, 4.4, 1.3], [ 0, 1, 0]],
        [[6.4, 3.1, 5.5, 1.8], [ 0, 0, 1]],
        [[4.4, 3.0, 1.3, 0.2], [ 1, 0, 0]],
        [[5.6, 3.0, 4.1, 1.3], [ 0, 1, 0]],
        [[6.0, 3.0, 4.8, 1.8], [ 0, 0, 1]],
        [[5.1, 3.4, 1.5, 0.2], [ 1, 0, 0]],
        [[5.5, 2.5, 4.0, 1.3], [ 0, 1, 0]],
        [[6.9, 3.1, 5.4, 2.1], [ 0, 0, 1]],
        [[5.0, 3.5, 1.3, 0.3], [ 1, 0, 0]],
        [[5.5, 2.6, 4.4, 1.2], [ 0, 1, 0]],
        [[6.7, 3.1, 5.6, 2.4], [ 0, 0, 1]],
        [[4.5, 2.3, 1.3, 0.3], [ 1, 0, 0]],
        [[6.1, 3.0, 4.6, 1.4], [ 0, 1, 0]],
        [[6.9, 3.1, 5.1, 2.3], [ 0, 0, 1]],
        [[4.4, 3.2, 1.3, 0.2], [ 1, 0, 0]],
        [[5.8, 2.6, 4.0, 1.2], [ 0, 1, 0]],
        [[5.8, 2.7, 5.1, 1.9], [ 0, 0, 1]],
        [[5.0, 3.5, 1.6, 0.6], [ 1, 0, 0]],
        [[5.0, 2.3, 3.3, 1.0], [ 0, 1, 0]],
        [[6.8, 3.2, 5.9, 2.3], [ 0, 0, 1]],
        [[5.1, 3.8, 1.9, 0.4], [ 1, 0, 0]],
        [[5.6, 2.7, 4.2, 1.3], [ 0, 1, 0]],
        [[6.7, 3.3, 5.7, 2.5], [ 0, 0, 1]],
        [[4.8, 3.0, 1.4, 0.3], [ 1, 0, 0]],
        [[5.7, 3.0, 4.2, 1.2], [ 0, 1, 0]],
        [[6.7, 3.0, 5.2, 2.3], [ 0, 0, 1]],
        [[5.1, 3.8, 1.6, 0.2], [ 1, 0, 0]],
        [[5.7, 2.9, 4.2, 1.3], [ 0, 1, 0]],
        [[6.3, 2.5, 5.0, 1.9], [ 0, 0, 1]],
        [[4.6, 3.2, 1.4, 0.2], [ 1, 0, 0]],
        [[6.2, 2.9, 4.3, 1.3], [ 0, 1, 0]],
        [[6.5, 3.0, 5.2, 2.0], [ 0, 0, 1]],
        [[5.3, 3.7, 1.5, 0.2], [ 1, 0, 0]],
        [[5.1, 2.5, 3.0, 1.1], [ 0, 1, 0]],
        [[6.2, 3.4, 5.4, 2.3], [ 0, 0, 1]],
        [[5.0, 3.3, 1.4, 0.2], [ 1, 0, 0]],
        [[5.7, 2.8, 4.1, 1.3], [ 0, 1, 0]],
        [[5.9, 3.0, 5.1, 1.8], [ 0, 0, 1]],
        [[5.1, 3.5, 1.4, 0.2], [ 1, 0, 0]],
        [[7.0, 3.2, 4.7, 1.4], [ 0, 1, 0]],
        [[6.3, 3.3, 6.0, 2.5], [ 0, 0, 1]],
        [[4.9, 3.0, 1.4, 0.2], [ 1, 0, 0]],
        [[6.4, 3.2, 4.5, 1.5], [ 0, 1, 0]],
        [[5.8, 2.7, 5.1, 1.9], [ 0, 0, 1]],
        [[4.7, 3.2, 1.3, 0.2], [ 1, 0, 0]],
        [[6.9, 3.1, 4.9, 1.5], [ 0, 1, 0]],
        [[7.1, 3.0, 5.9, 2.1], [ 0, 0, 1]],
        [[4.6, 3.1, 1.5, 0.2], [ 1, 0, 0]],
        [[5.5, 2.3, 4.0, 1.3], [ 0, 1, 0]],
        [[6.3, 2.9, 5.6, 1.8], [ 0, 0, 1]],
        [[5.0, 3.6, 1.4, 0.2], [ 1, 0, 0]],
        [[6.5, 2.8, 4.6, 1.5], [ 0, 1, 0]],
        [[6.5, 3.0, 5.8, 2.2], [ 0, 0, 1]],
        [[5.4, 3.9, 1.7, 0.4], [ 1, 0, 0]],
        [[5.7, 2.8, 4.5, 1.3], [ 0, 1, 0]],
        [[7.6, 3.0, 6.6, 2.1], [ 0, 0, 1]],
        [[4.6, 3.4, 1.4, 0.3], [ 1, 0, 0]],
        [[6.3, 3.3, 4.7, 1.6], [ 0, 1, 0]],
        [[4.9, 2.5, 4.5, 1.7], [ 0, 0, 1]],
        [[5.0, 3.4, 1.5, 0.2], [ 1, 0, 0]],
        [[4.9, 2.4, 3.3, 1.0], [ 0, 1, 0]],
        [[7.3, 2.9, 6.3, 1.8], [ 0, 0, 1]],
        [[4.4, 2.9, 1.4, 0.2], [ 1, 0, 0]],
        [[6.6, 2.9, 4.6, 1.3], [ 0, 1, 0]],
        [[6.7, 2.5, 5.8, 1.8], [ 0, 0, 1]],
        [[4.9, 3.1, 1.5, 0.1], [ 1, 0, 0]],
        [[5.2, 2.7, 3.9, 1.4], [ 0, 1, 0]],
        [[7.2, 3.6, 6.1, 2.5], [ 0, 0, 1]],
        [[5.4, 3.7, 1.5, 0.2], [ 1, 0, 0]],
        [[5.0, 2.0, 3.5, 1.0], [ 0, 1, 0]],
        [[6.5, 3.2, 5.1, 2.0], [ 0, 0, 1]],
        [[4.8, 3.4, 1.6, 0.2], [ 1, 0, 0]],
        [[5.9, 3.0, 4.2, 1.5], [ 0, 1, 0]],
        [[6.4, 2.7, 5.3, 1.9], [ 0, 0, 1]],
        [[4.8, 3.0, 1.4, 0.1], [ 1, 0, 0]],
        [[6.0, 2.2, 4.0, 1.0], [ 0, 1, 0]],
        [[6.8, 3.0, 5.5, 2.1], [ 0, 0, 1]],
        [[4.3, 3.0, 1.1, 0.1], [ 1, 0, 0]],
        [[6.1, 2.9, 4.7, 1.4], [ 0, 1, 0]],
        [[5.7, 2.5, 5.0, 2.0], [ 0, 0, 1]],
        [[5.8, 4.0, 1.2, 0.2], [ 1, 0, 0]],
        [[5.6, 2.9, 3.6, 1.3], [ 0, 1, 0]],
        [[5.8, 2.8, 5.1, 2.4], [ 0, 0, 1]],
        [[5.7, 4.4, 1.5, 0.4], [ 1, 0, 0]],
        [[6.7, 3.1, 4.4, 1.4], [ 0, 1, 0]],
        [[6.4, 3.2, 5.3, 2.3], [ 0, 0, 1]],
        [[5.4, 3.9, 1.3, 0.4], [ 1, 0, 0]],
        [[5.6, 3.0, 4.5, 1.5], [ 0, 1, 0]],
        [[6.5, 3.0, 5.5, 1.8], [ 0, 0, 1]],
        [[5.1, 3.5, 1.4, 0.3], [ 1, 0, 0]],
        [[5.8, 2.7, 4.1, 1.0], [ 0, 1, 0]],
        [[7.7, 3.8, 6.7, 2.2], [ 0, 0, 1]],
        [[5.7, 3.8, 1.7, 0.3], [ 1, 0, 0]],
        [[5.2, 2.2, 4.5, 1.5], [ 0, 1, 0]],
        [[7.7, 2.6, 6.9, 2.3], [ 0, 0, 1]],
        [[5.1, 3.8, 1.5, 0.3], [ 1, 0, 0]],
        [[5.6, 2.5, 3.9, 1.1], [ 0, 1, 0]],
        [[6.0, 2.2, 5.0, 1.5], [ 0, 0, 1]],
        [[5.4, 3.4, 1.7, 0.2], [ 1, 0, 0]],
        [[5.9, 3.2, 4.8, 1.8], [ 0, 1, 0]],
        [[6.9, 3.2, 5.7, 2.3], [ 0, 0, 1]],
        [[5.1, 3.7, 1.5, 0.4], [ 1, 0, 0]],
        [[6.1, 2.8, 4.0, 1.3], [ 0, 1, 0]]
    ]

    # 25 dados para teste da rede (4 entradas/ 3 saídas)
    teste = [
        [[7.2, 3.0, 5.8, 1.6], [0, 0, 1]],
        [[5.2, 3.5, 1.5, 0.2], [1, 0, 0]],
        [[6.7, 3.0, 5.0, 1.7], [0, 1, 0]],
        [[6.1, 3.0, 4.9, 1.8], [0, 0, 1]],
        [[5.2, 3.4, 1.4, 0.2], [1, 0, 0]],
        [[6.0, 2.9, 4.5, 1.5], [0, 1, 0]],
        [[6.4, 2.8, 5.6, 2.1], [0, 0, 1]],
        [[4.7, 3.2, 1.6, 0.2], [1, 0, 0]],
        [[5.7, 2.6, 3.5, 1.0], [0, 1, 0]],
        [[5.6, 2.8, 4.9, 2.0], [0, 0, 1]],
        [[4.6, 3.6, 1.0, 0.2], [1, 0, 0]],
        [[6.3, 2.5, 4.9, 1.5], [0, 1, 0]],
        [[7.7, 2.8, 6.7, 2.0], [0, 0, 1]],
        [[5.1, 3.3, 1.7, 0.5], [1, 0, 0]],
        [[6.1, 2.8, 4.7, 1.2], [0, 1, 0]],
        [[6.3, 2.7, 4.9, 1.8], [0, 0, 1]],
        [[4.8, 3.4, 1.9, 0.2], [1, 0, 0]],
        [[6.4, 2.9, 4.3, 1.3], [0, 1, 0]],
        [[6.7, 3.3, 5.7, 2.1], [0, 0, 1]],
        [[5.0, 3.0, 1.6, 0.2], [1, 0, 0]],
        [[6.6, 3.0, 4.4, 1.4], [0, 1, 0]],
        [[7.2, 3.2, 6.0, 1.8], [0, 0, 1]],
        [[5.0, 3.4, 1.6, 0.4], [1, 0, 0]],
        [[5.8, 2.8, 4.8, 1.4], [0, 1, 0]],
        [[6.2, 2.8, 4.8, 1.8], [0, 0, 1]]
    ]

    '''
    #base = datasets.load_iris()

    #print (base.data)
    #print(base.target)

    entradas = base.data
    saidas = base.target

    #x = 1
    #print(numpy.exp(-x))

    #print('entradas')
    #print(entradas)

    #print('saídas')
    #print(saidas)

    final = []
    aux = []
    #for i, entrada in enumerate(entradas):
    #    aux = []
    #    aux.append(entrada)
    #    aux.append([saidas[i]])
    #    final.append(aux)

    #print (final)

    #print(entradas[0].size)
    '''

    # constrói rede neural
    # var1: quantidade de neurônios na camada inicio (entrada)
    # var2: quantidade de neurônios na camada oculta
    # var3: quantidade de neurônios na ultima camada (saída)
    # var4: máximo de iterações
    # var5: taxa de aprendizagem
    # var6: momento
    # não é necessário a inclusão do bias, o mesmo já é incluso na costrunção da rede
    rede = RedeNeural(4, 7, 3, 5000, 0.15, 0.1)

    # treinar rede
    criar_linha()
    rede.treinar(flor_lis)
    print("Rede treinada!")

    # testar
    # resultados:   1 , 0 , 0 -> Setosa
    #               0 , 1 , 0 -> Versicolor
    #               0 , 0 , 1 -> Virginia
    criar_linha()
    rede.test(teste)


if __name__ == '__main__':
    iniciar()

