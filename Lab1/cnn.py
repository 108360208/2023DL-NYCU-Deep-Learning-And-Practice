import numpy as np

def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs, labels = [], []

    for pt in pts:
        inputs.append([pt[0], pt[1]])
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)

# 定義Sigmoid激活函數及其導數
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# 定義捲積層
class ConvLayer:
    def __init__(self, input_shape, num_filters, kernel_size):
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.weights = np.random.randn(num_filters, kernel_size, kernel_size)
        self.bias = np.zeros((num_filters, 1))
        self.output_shape = (input_shape[0] - kernel_size + 1, input_shape[1] - kernel_size + 1, num_filters)

    def forward(self, inputs):
        self.inputs = inputs
        self.padded_inputs = np.pad(inputs, ((0, 0), (self.kernel_size - 1, self.kernel_size - 1), (self.kernel_size - 1, self.kernel_size - 1)), mode='constant')
        self.outputs = np.zeros(self.output_shape)

        for f in range(self.num_filters):
            for i in range(self.output_shape[0]):
                for j in range(self.output_shape[1]):
                    receptive_field = self.padded_inputs[:, i:i+self.kernel_size, j:j+self.kernel_size]
                    self.outputs[i, j, f] = np.sum(receptive_field * self.weights[f]) + self.bias[f]

        return self.outputs

    def backward(self, d_outputs, learning_rate):
        d_weights = np.zeros_like(self.weights)
        d_bias = np.zeros_like(self.bias)
        d_inputs = np.zeros(self.input_shape)

        for f in range(self.num_filters):
            for i in range(self.output_shape[0]):
                for j in range(self.output_shape[1]):
                    receptive_field = self.padded_inputs[:, i:i+self.kernel_size, j:j+self.kernel_size]
                    d_weights[f] += d_outputs[i, j, f] * receptive_field
                    d_bias[f] += d_outputs[i, j, f]

                    d_input_slice = d_outputs[i, j, f] * self.weights[f]
                    d_inputs[:, i:i+self.kernel_size, j:j+self.kernel_size] += d_input_slice

        self.weights -= learning_rate * d_weights
        self.bias -= learning_rate * d_bias

        return d_inputs

# 定義池化層 (Max Pooling)
class PoolingLayer:
    def __init__(self, pool_size):
        self.pool_size = pool_size

    def forward(self, inputs):
        self.inputs = inputs
        batch_size, input_height, input_width, num_channels = inputs.shape
        output_height = input_height // self.pool_size
        output_width = input_width // self.pool_size
        self.outputs = np.zeros((batch_size, output_height, output_width, num_channels))

        for b in range(batch_size):
            for c in range(num_channels):
                for i in range(output_height):
                    for j in range(output_width):
                        self.outputs[b, i, j, c] = np.max(inputs[b, i*self.pool_size:(i+1)*self.pool_size, j*self.pool_size:(j+1)*self.pool_size, c])

        return self.outputs

    def backward(self, d_outputs):
        d_inputs = np.zeros_like(self.inputs)
        batch_size, output_height, output_width, num_channels = d_outputs.shape

        for b in range(batch_size):
            for c in range(num_channels):
                for i in range(output_height):
                    for j in range(output_width):
                        pool_slice = self.inputs[b, i*self.pool_size:(i+1)*self.pool_size, j*self.pool_size:(j+1)*self.pool_size, c]
                        max_val = np.max(pool_slice)
                        mask = (pool_slice == max_val)
                        d_inputs[b, i*self.pool_size:(i+1)*self.pool_size, j*self.pool_size:(j+1)*self.pool_size, c] = d_outputs[b, i, j, c] * mask

        return d_inputs

# 定義全連接層 (只有一個神經元，作為二元分類)
class DenseLayer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.zeros((1, output_size))

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weights) + self.bias
        return sigmoid(self.outputs)

    def backward(self, d_outputs, learning_rate):
        d_inputs = np.dot(d_outputs, self.weights.T)
        d_weights = np.dot(self.inputs.T, d_outputs)
        d_bias = np.sum(d_outputs, axis=0, keepdims=True)

        self.weights -= learning_rate * d_weights
        self.bias -= learning_rate * d_bias

        return d_inputs

# 定義簡單的CNN模型
class SimpleCNN:
    def __init__(self):
        self.conv_layer = ConvLayer(input_shape=(28, 28, 1), num_filters=8, kernel_size=3)
        self.pool_layer = PoolingLayer(pool_size=2)
        self.dense_layer = DenseLayer(input_size=13*13*8, output_size=1)

    def forward(self, inputs):
        conv_output = self.conv_layer.forward(inputs)
        pool_output = self.pool_layer.forward(conv_output)
        pool_output_flattened = pool_output.reshape(pool_output.shape[0], -1)
        output = self.dense_layer.forward(pool_output_flattened)
        return output

    def backward(self, d_outputs, learning_rate):
        d_dense = self.dense_layer.backward(d_outputs, learning_rate)
        d_pooled = d_dense.reshape(d_dense.shape[0], self.pool_layer.output_shape[0], self.pool_layer.output_shape[1], self.conv_layer.num_filters)
        d_conv = self.pool_layer.backward(d_pooled)
        d_inputs = self.conv_layer.backward(d_conv, learning_rate)
        return d_inputs

# 定義訓練函數
def train(model, inputs, labels, epochs, learning_rate):
    for epoch in range(epochs):
        total_loss = 0

        for i in range(len(inputs)):
            input_data = inputs[i:i+1]
            label = labels[i:i+1]

            # Forward pass
            output = model.forward(input_data)

            # Compute loss (binary cross-entropy)
            loss = -label * np.log(output) - (1 - label) * np.log(1 - output)
            total_loss += loss

            # Backward pass
            d_output = (output - label) / (output * (1 - output))
            model.backward(d_output, learning_rate)

        total_loss /= len(inputs)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss[0][0]}")

# 產生訓練資料
inputs, labels = generate_linear(n=100)

# 將資料reshape成CNN輸入的形狀 (這邊假設輸入為28x28的圖片，但實際上資料是一個點的座標)
inputs = inputs.reshape(-1, 28, 28, 1)

# 建立並訓練模型
model = SimpleCNN()
train(model, inputs, labels, epochs=50, learning_rate=0.1)