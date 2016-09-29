import tensorflow as tf
import numpy as np
from hw0 import read_images, read_labels
import os

class Net(object):
    def __init__(self, input_size, output_size, learn_rate=0.5, dir_path=None):
        input = tf.placeholder(tf.float32, [None, input_size], name='input')
        weights = tf.Variable(tf.zeros([input_size, output_size]))
        bias = tf.Variable(tf.zeros([output_size]))
        output = tf.nn.softmax(tf.matmul(input, weights) + bias)
        desired_output = tf.placeholder(tf.float32, [None, output_size], name='desired_output')
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(desired_output * tf.log(output), reduction_indices=[1]))
        train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(cross_entropy)
        
        self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())
        self.input = input
        self.output = output
        self.desired_output = desired_output
        self.train_step = train_step
        
        self.weights = weights
        self.bias = bias
        
        if dir_path[-1] != '/': dir_path += '/'
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        self.dir_path = dir_path
        self.was_restored = False
        if dir_path:
            self.saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(self.dir_path)
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(self.session, ckpt.model_checkpoint_path)
                self.was_restored = True
        else:
            self.saver = None
        
    def train(self, inputs, outputs):
        self.session.run(self.train_step, feed_dict={self.input: inputs, self.desired_output: outputs})
    
    def classify(self, inputs):
        return self.session.run(tf.argmax(self.output, 1), feed_dict={self.input: inputs})
    
    def save(self, step):
        self.saver.save(self.session, self.dir_path + 'model.ckpt', global_step=step)
    
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

def train(iterations=1000, train=True):
    net = Net(784, 10, dir_path='save/hw2-2')
    print 'RESTORED: {0}'.format(net.was_restored)
    test_in, test_out = load_dataset('t10k')
    train_in, train_out = load_dataset('train')
    print "Accuracy: {0}".format(net.evaluate(train_in, train_out))
    if train:
        for i in xrange(iterations):
            _in, _out = random_batch(train_in, train_out)
            assert _in.shape[0] == 100 and _in.shape[1] == 784 and _out.shape[1] == 10
            net.train(_in, _out)
            print "Accuracy: {0}".format(net.evaluate(train_in, train_out))
            if i % 10 == 0:
                print 'Saving'
                net.save(i)
    return net

if __name__ == '__main__':
    # simple_train()
    train(train=False)
