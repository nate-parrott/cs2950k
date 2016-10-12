from net import Net, batch_generator
import tensorflow as tf
import numpy as np
from util import rand_variable

class LanguageModel(Net):
    vocab_size = None
    ngrams = 2
    
    def setup(self):
        assert self.vocab_size
        self.input = tf.placeholder(tf.int32, [None, self.ngrams])
        self.desired_output = tf.placeholder(tf.int32, [None])
        input_dense = tf.one_hot(self.input, self.vocab_size)
        
        embedding_size = 30
        embedding = rand_variable([self.vocab_size, embedding_size])
        word_vecs = tf.unpack(input_dense, axis=1) # array of N tensors [None, vocab_size] each containing a one-hot representation of the nth word
        embedded_vec = tf.concat(1, [tf.matmul(word_vec, embedding) for word_vec in word_vecs])
        # assert len(embedded_vec.shape) == 2 and embedded_vec[1] == self.ngrams * self.vocab_size
        with tf.control_dependencies([tf.assert_equal(embedded_vec.get_shape()[1], self.ngrams * embedding_size)]):
            layer_size = embedding_size * self.ngrams
            w1 = rand_variable((self.ngrams * embedding_size, layer_size))
            b1 = rand_variable((layer_size,))
            relu = tf.nn.relu(tf.matmul(embedded_vec, w1) + b1)
            w2 = rand_variable((layer_size, self.vocab_size))
            b2 = rand_variable((self.vocab_size,))
            self.output = tf.nn.softmax(tf.matmul(relu, w2) + b2)
        
        desired_output_dense = tf.one_hot(self.desired_output, self.vocab_size)
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(desired_output_dense * tf.log(self.output), reduction_indices=[1]))
        self.learn_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        
        correct = tf.equal(tf.argmax(self.output, 1), tf.argmax(desired_output_dense, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        likelihoods_of_correct_answers = tf.reduce_sum(tf.mul(desired_output_dense, self.output), reduction_indices=[1])
        self.perplexity = tf.exp(tf.reduce_mean(-tf.log(likelihoods_of_correct_answers)))
    
    def train(self, inputs, outputs):
        self.session.run(self.learn_step, feed_dict={self.input: inputs, self.desired_output: outputs})
    
    def evaluate(self, inputs, outputs):
        accuracy, perplexity = self.session.run((self.accuracy, self.perplexity), feed_dict={self.input: inputs, self.desired_output: outputs})
        print "Perplexity:", perplexity
        return accuracy
    
    def predict(self, inputs):
        return self.session.run(self.output, feed_dict={self.input: inputs})
        
NOTHING = "*NOTHING*"
STOP = "STOP"

def load_txt(name, ngrams=2):
    grams = []
    for line in open(name):
        words = [NOTHING] * ngrams + line.strip().split()
        for i in xrange(ngrams, len(words)):
            ngram = words[i-ngrams:i]
            word = words[i]
            grams.append((ngram, word))
    return grams

# def vectorize_sparse(grams, vocab_lookup):
#     # grams = grams[:min(len(grams), 1000)] # REMOVE ME
#     ngrams = len(grams[0][0])
#     vocab_size = len(vocab_lookup)
#     input_vector = np.zeros((len(grams), ngrams, vocab_size), dtype=np.bool_)
#     output_vector = np.zeros((len(grams), vocab_size), dtype=np.bool_)
#     for i, (ngram, word) in enumerate(grams):
#         for j, w in enumerate(ngram):
#             input_vector[i][j][vocab_lookup[w]] = 1
#         output_vector[i][vocab_lookup[w]] = 1
#     return input_vector, output_vector

def vectorize(grams, vocab_lookup):
    ngrams = len(grams[0][0])
    vocab_size = len(vocab_lookup)
    input_vector = np.zeros((len(grams), ngrams), np.int)
    output_vector = np.zeros((len(grams),), np.int)
    for i, (ngram, word) in enumerate(grams):
        for j, w in enumerate(ngram):
            input_vector[i][j] = vocab_lookup[w]
        output_vector[i] = vocab_lookup[word]
    return input_vector, output_vector

def generate_sentence(net, vocab_lookup):
    reverse_vocab_lookup = dict((v,k) for (k,v) in vocab_lookup.iteritems())
    prev = [NOTHING, NOTHING]
    while prev[-1] != STOP:
        vec = np.zeros((1, 2), np.int)
        vec[0][0] = vocab_lookup[prev[-2]]
        vec[0][1] = vocab_lookup[prev[-1]]
        word_probs = net.predict(vec)[0]
        word_idx = np.random.choice(net.vocab_size, p=word_probs)
        prev.append(reverse_vocab_lookup[word_idx])
    return ' '.join(prev[2:-1])

def run(text_gen=False):
    train = load_txt('train.txt')
    test = load_txt('test.txt')
    
    vocab = list(set(gram[1] for gram in test + train)) + [NOTHING]
    vocab_lookup = {}
    for i, word in enumerate(vocab):
        vocab_lookup[word] = i
    
    print 'Starting vectorizing'
    train_matrix_in, train_matrix_out = vectorize(train, vocab_lookup)
    test_matrix_in, test_matrix_out = vectorize(test, vocab_lookup)
    print 'Done vectorizing'
    
    net = LanguageModel(dir_path='save/hw4', vocab_size=len(vocab_lookup))
    #     def training_loop(self, training_inputs, training_outputs, testing_inputs, testing_outputs, training_batch_size=50, testing_batch_size=1000):
    if text_gen:
        while True:
            print generate_sentence(net, vocab_lookup)
            raw_input("...")
    else:
        train_batcher = batch_generator(train_matrix_in, train_matrix_out, size=20, epochs=1, print_progress=True)
        test_batcher = batch_generator(test_matrix_in, test_matrix_out, size=2000, random=True)
        net.training_loop(train_batcher, test_batcher, evaluation_interval=50)

if __name__ == '__main__':
    run(text_gen=False)
