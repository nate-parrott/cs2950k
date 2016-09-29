import tensorflow as tf
import numpy as np
from hw0 import read_images, read_labels

class Net(object):
    def __init__(self, input_width, input_height, output_size, learn_rate=1.0e-4):
        input = tf.placeholder(tf.float32, [None, input_width, input_height], name='input')
        input_reshaped = tf.reshape(input, [-1, input_width, input_height, 1])
        
        patch_size = 5
        conv1_feature_count = 32
        dropout_keep_prob = tf.placeholder(tf.float32)
        
        def create_conv(input, in_channels, out_channels):
            weights = weight_var([patch_size, patch_size, in_channels, out_channels])
            biases = weight_var([out_channels])
            conv = tf.nn.conv2d(input, weights, strides=[1,1,1,1], padding='SAME')
            activation = tf.nn.relu(conv + biases)
            pooled = tf.nn.max_pool(activation, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
            return pooled
        
        def create_dense(input, input_size, output_size, relu=True):
            weights = weight_var([input_size, output_size])
            biases = weight_var([output_size])
            r = tf.matmul(input, weights) + biases
            return tf.nn.relu(r) if relu else r
        
        conv1_output = create_conv(input_reshaped, 1, 32)
        conv2_output = create_conv(conv1_output, 32, 64)
        image_feature_count = (input_width / 4) * (input_height / 4) * 64
        flattened = tf.reshape(conv2_output, [-1, image_feature_count])
        # divide input_width and input_height by 4 because we cut the image in half during max_pooing in create_conv
        dense = create_dense(flattened, image_feature_count, 1024, relu=True)
        dense_dropout = tf.nn.dropout(dense, dropout_keep_prob)
        output = tf.nn.softmax(create_dense(dense_dropout, 1024, output_size))
        desired_output = tf.placeholder(tf.float32, [None, output_size], name='desired_output')
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(desired_output * tf.log(output), reduction_indices=[1]))
        
        train_step = tf.train.AdamOptimizer(learn_rate).minimize(cross_entropy)
        session = tf.Session()
        session.run(tf.initialize_all_variables())
        
        self.input = input
        self.output = output
        self.desired_output = desired_output
        self.dropout_keep_prob = dropout_keep_prob
        self.train_step = train_step
        self.session = session
    
    def train(self, inputs, outputs):
        self.session.run(self.train_step, feed_dict={self.input: inputs, self.desired_output: outputs, self.dropout_keep_prob: 0.5})
    
    def evaluate(self, inputs, outputs):
        correct = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.desired_output, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        accuracy_val = self.session.run(accuracy, feed_dict={self.input: inputs, self.desired_output: outputs, self.dropout_keep_prob: 1})
        return int(round(accuracy_val * 100))

def weight_var(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def one_hot(idx, size):
    v = np.zeros(size)
    v[idx] = 1
    return v

def load_dataset(name):
    def featurize(image):
        return image.astype(float) / 255.0
    images = np.array([featurize(image) for image in read_images(name + '-images.idx3-ubyte')])
    labels = np.array([one_hot(label, 10) for label in read_labels(name + '-labels.idx1-ubyte')])
    return images, labels

def random_batch(inputs, outputs, count=50):
    indices = np.random.randint(0, len(inputs)-1, count)
    return inputs.take(indices, axis=0), outputs.take(indices, axis=0)

def train():
    net = Net(28, 28, 10)
    test_in, test_out = load_dataset('t10k')
    train_in, train_out = load_dataset('train')
    # print test_in.shape
    # print 'loaded'
    evaluate_every = 50
    for i in xrange(2000):
        _in, _out = random_batch(train_in, train_out)
        # print 'got batch'
        net.train(_in, _out)
        # print 'trained'
        # quit()
        if i % evaluate_every == 0:
            _in, _out = random_batch(test_in, test_out, count=2000)
            # print 'testing'
            # _in, _out = train_in, train_out
            print "Accuracy: {0}".format(net.evaluate(_in, _out))

if __name__ == '__main__':
    # simple_train()
    train()
