from hw2 import train
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import sys

def backtrack(digit):
    net = train(train=False)
    
    bias = tf.constant(net.session.run(net.bias.value()))
    weights = tf.constant(net.session.run(net.weights.value()))
    
    input = tf.Variable(tf.truncated_normal([1,784], stddev=0.1, mean=0.2))
    output = tf.nn.softmax(tf.matmul(input, weights) + bias)
    desired_output = tf.placeholder(tf.float32, [11], name='desired_output')
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(desired_output * tf.log(output), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    for _ in range(1000):
        d = {desired_output: one_hot(8, 11)}
        sess.run(train_step, feed_dict=d)
        print sess.run(cross_entropy, feed_dict=d)
    four = sess.run(input.value())
    print np.reshape(four, (28,28))*255
    # print net.classify(four)
    # open('save/four.json', 'w').write(json.dumps(np.reshape(four, (28,28)).tolist()))
    show_image(np.reshape(four, (28,28)))
    
def one_hot(idx, size):
    v = np.zeros(size)
    v[idx] = 1
    return v

def show_image(nparray):
    im = Image.fromarray(np.uint8(nparray * 255))
    im.show()

if __name__ == '__main__':
    backtrack(digit=int(sys.argv[1]))
