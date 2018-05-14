import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

import network

input_units  = 784
hidden_units = 30
output_units = 10

net = network.Network([input_units, hidden_units, output_units])

learning_rate   = 3.0
epochs          = 30
mini_batch_size = 10
net.SGD(training_data, 30, 10, learning_rate, test_data=test_data)