from hw0 import read_images, read_labels
import numpy as np
import random

def softmax(vec):
    evec = np.exp(vec)
    return evec / np.sum(evec)

class Net(object):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.zeros([input_size, output_size])
        self.bias = np.zeros([output_size])
    
    def forward(self, input):
        assert len(input.shape) == 1 and input.shape[0] == self.input_size
        hidden_layer = np.dot(input, self.weights) + self.bias
        assert len(hidden_layer.shape) == 1 and hidden_layer.shape[0] == self.output_size
        return hidden_layer, softmax(hidden_layer)
    
    def predict_label(self, input):
        _, probs = self.forward(input)
        return max(xrange(self.output_size), key=lambda i: probs[i])
    
    def train(self, label, input, learn_rate):
        hidden_layer, probs = self.forward(input)
        
        loss = -np.log(probs[label])
        
        # def hidden_layer_loss_derivative(index):
        #     if index == label:
        #         return 1 - probs[index]
        #     else:
        #         return -probs[index]
        # hidden_layer_error_derivatives = np.array([hidden_layer_loss_derivative(i) for i in xrange(self.output_size)])
        hidden_layer_error_derivatives = -probs
        hidden_layer_error_derivatives[label] += 1
        
        # print input.shape
        # print hidden_layer_error_derivatives.shape
        weight_error_derivatives = np.outer(input, hidden_layer_error_derivatives)
        
        self.bias += hidden_layer_error_derivatives * learn_rate
        self.weights += weight_error_derivatives * learn_rate
    
def featurize(image):
    return np.ndarray.flatten(image) / 255.0

# def onehot(label):
#     output = np.zero(10)
#     output[label] = 1
#     return output

def load_dataset(name):
    labels = read_labels(name + '-labels.idx1-ubyte')
    def featurize(image):
        return np.array.flatten(image).astype(float) / 255.0
    images = np.array([featurize(image) for image in read_images(name + '-images.idx3-ubyte')])
    return list(zip(labels, images))

def train(train_data, test_data):
    train_data = list(train_data)
    max_label = max((label for label, _ in train_data))
    net = Net(len(train_data[0][1]), max_label + 1)
    while True:
        right = 0
        wrong = 0
        for label, input in test_data:
            if net.predict_label(input) == label:
                right += 1
            else:
                wrong += 1
        print "{0}% correct".format(int(round(right * 100.0 / (right + wrong))))
        
        random.shuffle(train_data) 
        for label, input in train_data:
            net.train(label, input, 0.5)

def simple_train():
    SIMPLE_DATASET = [
            (0, [1,0,0]),
            (1, [0,1,0]),
            (2, [0,0,1]),
            (3, [1,1,1]),
            (4, [0,0,0])
        ]
    SIMPLE_DATASET = [(label, np.array(vec)) for label, vec in SIMPLE_DATASET]
    train(SIMPLE_DATASET, SIMPLE_DATASET)

def train_images():
    train(load_dataset('train'), load_dataset('t10k'))

if __name__ == '__main__':
    train_images()
    # simple_train()
