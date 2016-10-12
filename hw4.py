from net import Net, batch_generator
import tensorflow as tf
import numpy as np
from util import rand_variable

class LanguageModel(Net):
    vocab_size = None
    
    def setup(self):
        assert self.vocab_size
        self.input = tf.placeholder(tf.int32, [None])
        self.desired_output = tf.placeholder(tf.int32, [None])
        
        embedding_size = 30
        embedding = rand_variable([self.vocab_size, embedding_size])        
        embedded_vec = tf.nn.embedding_lookup(embedding, self.input)
        
        layer_size = 100
        w1 = rand_variable((embedding_size, layer_size))
        b1 = rand_variable((layer_size,))
        relu = tf.nn.relu(tf.matmul(embedded_vec, w1) + b1)
        w2 = rand_variable((layer_size, self.vocab_size))
        b2 = rand_variable((self.vocab_size,))
        self.logits = tf.matmul(relu, w2) + b2
        self.output = tf.nn.softmax(self.logits)
        
        # cross_entropy = tf.reduce_mean(-tf.reduce_sum(desired_output_dense * tf.log(self.output), reduction_indices=[1]))
        cross_entropies = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, self.desired_output)
        cross_entropy = tf.reduce_mean(cross_entropies)
        self.learn_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        
        # correct = tf.equal(tf.argmax(self.output, 1), tf.argmax(desired_output_dense, 1))
        # self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        # likelihoods_of_correct_answers = tf.reduce_sum(tf.mul(desired_output_dense, self.output), reduction_indices=[1])
        self.perplexity = tf.exp(cross_entropy)
    
    def train(self, inputs, outputs):
        self.session.run(self.learn_step, feed_dict={self.input: inputs, self.desired_output: outputs})
    
    def evaluate(self, inputs, outputs):
        perplexity = self.session.run(self.perplexity, feed_dict={self.input: inputs, self.desired_output: outputs})
        print "Perplexity:", perplexity
        return 0
    
    def predict(self, inputs):
        return self.session.run(self.output, feed_dict={self.input: inputs})
        
NOTHING = "*NOTHING*"
STOP = "STOP"

def load_txt(name):
    grams = []
    for line in open(name):
        words = [NOTHING] + line.strip().split()
        for bigram in zip(words[:-1], words[1:]):
            grams.append(bigram)
    return grams

def generate_sentence(net, vocab_lookup):
    reverse_vocab_lookup = {v: k for k, v in vocab_lookup.iteritems()}
    words = [NOTHING]
    while words[-1] != 'STOP':
        word_probs = net.predict([vocab_lookup[words[-1]]])[0]
        word_idx = np.random.choice(net.vocab_size, p=word_probs)
        words.append(reverse_vocab_lookup[word_idx])
    return ' '.join(words[1:-1])

def vectorize(grams, vocab_lookup):
    # grams = grams[:1000] # REMOVE ME
    vocab_size = len(vocab_lookup)
    input_vector = np.zeros((len(grams),), np.int)
    output_vector = np.zeros((len(grams),), np.int)
    unk = vocab_lookup['*CUNK*'] # TODO: something better
    for i, (prev_word, word) in enumerate(grams):
        input_vector[i] = vocab_lookup.get(prev_word, unk)
        output_vector[i] = vocab_lookup.get(word, unk)
    return input_vector, output_vector

def run(text_gen=False, eval=False):
    train = load_txt('train.txt')
    test = load_txt('test.txt')
    
    vocab = list(set(gram[1] for gram in train)) + [NOTHING]
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
    elif eval:
        print net.evaluate(test_matrix_in, test_matrix_out)
    else:
        train_batcher = batch_generator(train_matrix_in, train_matrix_out, size=20, epochs=1, print_progress=True)
        test_batcher = batch_generator(test_matrix_in, test_matrix_out, size=8000, random=True)
        net.training_loop(train_batcher, test_batcher, evaluation_interval=500)

if __name__ == '__main__':
    run(text_gen=True)
