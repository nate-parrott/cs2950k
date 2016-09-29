import tensorflow as tf
import numpy as np
from hw0 import read_images, read_labels

class Net(object):
    def __init__(self, input_size, output_size, learn_rate=0.5):
        input = tf.placeholder(tf.float32, [None, input_size], name='input')
        weights = tf.Variable(tf.zeros([input_size, output_size]))
        bias = tf.Variable(tf.zeros([output_size]))
        output = tf.nn.softmax(tf.matmul(input, weights) + bias)
        desired_output = tf.placeholder(tf.float32, [None, output_size], name='desired_output')
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(desired_output * tf.log(output), reduction_indices=[1]))
        train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(cross_entropy)
        session = tf.Session()
        session.run(tf.initialize_all_variables())
        
        self.input = input
        self.output = output
        self.desired_output = desired_output
        self.train_step = train_step
        self.session = session
    
    def train(self, inputs, outputs):
        self.session.run(self.train_step, feed_dict={self.input: inputs, self.desired_output: outputs})
    
    def evaluate(self, inputs, outputs):
        correct = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.desired_output, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        accuracy_val = self.session.run(accuracy, feed_dict={self.input: inputs, self.desired_output: outputs})
        return int(round(accuracy_val * 100))

def one_hot(idx, size):
    v = np.zeros(size)
    v[idx] = 1
    return v

def load_dataset(name):
    def featurize(image):
        return image.flatten().astype(float) / 255.0
    images = np.array([featurize(image) for image in read_images(name + '-images.idx3-ubyte')])
    labels = np.array([one_hot(label, 10) for label in read_labels(name + '-labels.idx1-ubyte')])
    return images, labels

def simple_train():
    net = Net(3, 5)
    SIMPLE_DATASET = [
            (0, [1,0,0]),
            (1, [0,1,0]),
            (2, [0,0,1]),
            (3, [1,1,1]),
            (4, [0,0,0])
        ]
    inputs = np.array([np.array(vec) for _, vec in SIMPLE_DATASET])
    outputs = np.array([one_hot(label, 5) for label, _ in SIMPLE_DATASET])
    while 1:
        net.train(inputs, outputs)
        print "Accuracy: {0}".format(net.evaluate(inputs, outputs))

def random_batch(inputs, outputs, count=100):
    indices = np.random.randint(0, len(inputs)-1, count)
    return inputs.take(indices, axis=0), outputs.take(indices, axis=0)

def train(iterations=1000):
    net = Net(784, 10)
    test_in, test_out = load_dataset('t10k')
    train_in, train_out = load_dataset('train')
    for _ in xrange(iterations):
        _in, _out = random_batch(train_in, train_out)
        assert _in.shape[0] == 100 and _in.shape[1] == 784 and _out.shape[1] == 10
        net.train(_in, _out)
        print "Accuracy: {0}".format(net.evaluate(train_in, train_out))
    return net

if __name__ == '__main__':
    # simple_train()
    train()
