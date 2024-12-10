import numpy

class neuralNetwork:

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.lr = learningrate

         # Матрицы весовых коэффициентов связей wih (между входным и скрытым
        # слоями) и who (между скрытым и выходным слоями).
        # Весовые коэффициенты связей между узлом i и узлом j следующего слоя
        # обозначены как w_i_j:
        # wll w21
        # wl2 w22 и т.д.
        self.wih = (numpy.random.rand(self.hnodes, self.inodes)-0.5)
        self.who = (numpy.random.rand(self.onodes, self.hnodes)-0.5)

        pass

    def train():
        pass

    def query():
        pass

inputNodes = 3
hiddenNodes= 3
outputNodes = 3

learningRate = 0.3

n = neuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)

