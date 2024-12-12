import numpy, scipy.special, scipy.misc, time, aiogram
import os
from PIL import Image

start = time.time()

class neuralNetwork:

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # Матрицы весовых коэффициентов связей wih (между входным и скрытым
        # слоями) и who (между скрытым и выходным слоями).
        # Весовые коэффициенты связей между узлом i и узлом j следующего слоя
        # обозначены как w_i_j:
        # wll w21
        # wl2 w22 и т.д.ccc
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        self.lr = learningrate
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    def train(self, inputs_list, targets_list):
        # преобразовать список входных значений в двухмерный массив 
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        # рассчитать входящие сигналы для скрытого слоя 
        hidden_inputs = numpy.dot(self.wih, inputs)
        # рассчитать исходящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)
        # рассчитать входящие сигналы для выходного слоя
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # рассчитать исходящие сигналы для выходного слоя
        final_outputs = self.activation_function(final_inputs)

        # ошибка = целевое значение - фактическое значение
        output_errors = targets - final_outputs
        # ошибки скрытого слоя - это ошибки output_errors,
        # распределенные пропорционально весовым коэффициентам связей
        # и рекомбинированные на скрытых узлах
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # обновить весовые коэффициенты связей между скрытым и выходным слоями
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        # обновить весовые коэффициенты связей между входным и скрытым слоями
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))

        pass

    def query(self, inputs_list):
        # Сначала преобразуем список входных данных в двуерный массив
        inputs = numpy.array(inputs_list, ndmin=2).T

        # Рассчёт входящих сигналов для скрытого слоя
        hidden_inputs = numpy.dot(self.wih, inputs)
        # Рассчёт исходящих сигналов для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)

        # Рассчёт входящих сгналов для выходного слоя
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # Рассчёт исходящих сигналов для выходного слоя
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
        
inputNodes = 784
hiddenNodes= 500
outputNodes = 10
learningRate = 0.1

n = neuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)

# тестовый набор данных CSV-файла набора MNIST
training_data_file = open("mnist_dataset/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()


# тренировка нейронной сети


# перебрать все записи в тренировочном наборе данных
iterations = 5
for i in range(iterations):
    for record in training_data_list:
        # получить список значений, используя символы запятой (',') в качестве разделителей
        all_values = record.split(',')
        # масштабировать и сместить входные значения
        inputs = (numpy.asarray((all_values[1:]), dtype=float) / 255.0 * 0.99) + 0.01
        # создать целевые выходные значения (все равны 0,01, за исключением желаемого маркерного значения, равного 0,99)
        targets = numpy.zeros(outputNodes) + 0.01
        # all_values[0] - целевое маркерное значение для данной записи
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)

# загрузить в список тестовый набор данных CSV-файла набора MNIST 
test_data_file = open("mnist_dataset/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines() 
test_data_file.closed

# тестирование нейронной сети
# журнал оценок работы сети, первоначально пустой 
scorecard = []
# перебрать все записи в тестовом наборе данных 
for record in test_data_list:
    # получить список значений из записи, используя символы
    # запятой (*,1) в качестве разделителей 
    all_values = record.split(',')
    # правильный ответ - первое значение 
    correct_label = int(all_values[0]) 
    # масштабировать и сместить входные значения
    inputs = (numpy.asarray((all_values[1:]), dtype=float) / 255.0 * 0.99) + 0.01
    # опрос сети
    outputs = n.query(inputs)
    # индекс наибольшего значения является маркерным значением 
    label = numpy.argmax(outputs)
    # присоединить оценку ответа сети к концу списка 
    if (label == correct_label) :
        # в случае правильного ответа сети присоединить
        # к списку значение 1 
        scorecard.append(1)
    else:
        # в случае неправильного ответа сети присоединить
        # к списку значение 0 
        scorecard.append(0) 
scorecard_array = numpy.asarray(scorecard, dtype=float)
print ("эффективность = ", scorecard_array.sum() / len(scorecard))



def convert_and_resize_images(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".bmp"):
            # Открываем изображение
            img = Image.open(os.path.join(folder_path, filename))
            # Масштабируем до 28x28 пикселей
            img = img.resize((28, 28), Image.ANTIALIAS)
            # Конвертируем в PNG и сохраняем
            new_filename = os.path.splitext(filename)[0] + ".png"
            new_folder_path = ('/mnist_dataset/test_photo_12_12_png')
            img.save(os.path.join(new_folder_path, new_filename), "PNG")
convert_and_resize_images('/mnist_dataset/test_photo_12_12_jpg')





finish = time.time()
print(f"Время выполнения = {finish-start}")