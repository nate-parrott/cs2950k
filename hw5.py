from net import Net, batch_generator
import tensorflow as tf
import numpy as np
from util import rand_variable
from collections import defaultdict
import re
import random

UNK = "*UNK*"
START = "*START*"
END = "*END*"

NUM_STEPS = 20

class RNN(Net):
    num_steps = NUM_STEPS
    vocab_size = None
    batch_size = 100
    prev_state = None
    
    def setup(self):
        assert self.vocab_size
        
        input = tf.placeholder(tf.int32, [self.batch_size, self.num_steps])
        correct_output = tf.placeholder(tf.int32, [self.batch_size, self.num_steps])
        dropout_keep_prob = tf.placeholder(tf.float32)
        
        embedding_size = 50
        embedding = rand_variable([self.vocab_size, embedding_size])
        embedded = tf.nn.embedding_lookup(embedding, input)
        embedded = tf.nn.dropout(embedded, dropout_keep_prob)
        
        lstm_units = 256
        lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_units)
        initial_state = lstm.zero_state(self.batch_size, tf.float32)
        output, final_state = tf.nn.dynamic_rnn(lstm, embedded, dtype=tf.float32, initial_state=initial_state)
        output = tf.reshape(output, [self.batch_size * self.num_steps, lstm_units])
        
        w = rand_variable([lstm_units, self.vocab_size])
        bias = rand_variable([self.vocab_size])
        logits = tf.matmul(output, w) + bias
        
        correct_output_reshapes = tf.reshape(correct_output, [-1])
        seq_weights = tf.ones([self.batch_size * self.num_steps]) # the fuck is this
        loss_vec = tf.nn.seq2seq.sequence_loss_by_example([logits], [correct_output], [seq_weights])
        loss = tf.reduce_sum(loss_vec) / float(self.batch_size)
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
        
        self.input = input
        self.correct_output = correct_output
        self.output = output
        self.dropout_keep_prob = dropout_keep_prob
        self.initial_state = initial_state
        self.final_state = final_state
        self.train_step = train_step
        self.loss = loss
        self.perplexity = tf.exp(loss / float(self.num_steps))
    
    def train(self, inp, out):
        state = self.prev_state if self.prev_state is not None else self.session.run(self.initial_state)
        _, self.prev_state = self.session.run([self.train_step, self.final_state], feed_dict={self.input: inp, self.correct_output: out, self.dropout_keep_prob: 0.5, self.initial_state: state})
    
    def evaluate(self, inp, out):
        state = self.session.run(self.initial_state)
        loss, perplexity = self.session.run([self.loss, self.perplexity], feed_dict={self.input: inp, self.correct_output: out, self.dropout_keep_prob: 1, self.initial_state: state})
        print 'Loss: {0}, perplexity: {1}'.format(loss, perplexity)
        return loss

def flatten(arrays):
    return [item for sublist in arrays for item in sublist]

def to_mat(tokens, vocab_dict):
    padded_tokens = [START] * (NUM_STEPS - 2) + tokens + [END] * (NUM_STEPS - 2)
    padded_token_numbers = [vocab_dict[tk] for tk in padded_tokens]
    inp = np.zeros([len(padded_token_numbers), NUM_STEPS], np.int)
    out = np.zeros([len(padded_token_numbers), NUM_STEPS], np.int)
    for i, start in enumerate(xrange(len(padded_tokens) - NUM_STEPS - 1)):
        inp[i] = padded_token_numbers[start:start+NUM_STEPS]
        out[i] = padded_token_numbers[start+1:start+NUM_STEPS+1]
    return inp, out

def load_data(filename):
    lines = list(open(filename))
    random.shuffle(lines)
    
    def parse_sentence(line):
        line = line.lower()
        line = re.sub(r"([.,!?\"':;]+)", r" \1 ", line)
        return [START] + [tk for tk in line.split() if len(tk)] + [END]
    
    sentences = map(parse_sentence, lines)
    
    counts = defaultdict(int)
    for sentence in sentences:
        for word in sentence:
            counts[word] += 1
    
    words_by_count = [word for word, count in sorted(counts.items(), key=lambda (word,count): count, reverse=True)]
    # print len(words_by_count), 'words'
    
    top_words = sorted(set([UNK, START, END] + words_by_count[:min(len(words_by_count), 8000)]))
    vocab_dict = dict((word,i) for i, word in enumerate(top_words))
    
    def unk_sentence(sentence):
        return [(word if word in vocab_dict else UNK) for word in sentence]
    
    sentences = map(unk_sentence, sentences)
    split_idx = int(len(lines) * 0.9)
    
    def process_sentences(sents):
        return to_mat(flatten(sents), vocab_dict)
    
    train_in, train_out = process_sentences(sentences[:split_idx])
    test_in, test_out = process_sentences(sentences[split_idx:])
    return vocab_dict, train_in, train_out, test_in, test_out

def train():
    vocab_dict, train_in, train_out, test_in, test_out = load_data('trump.txt')
    n = RNN(dir_path='save/hw5', vocab_size=len(vocab_dict))
    train_batcher = batch_generator(train_in, train_out, size=n.batch_size, epochs=100)
    test_batcher = batch_generator(test_in, test_out, size=n.batch_size, random=True)
    n.training_loop(train_batcher, test_batcher, evaluation_interval=10)

if __name__ == '__main__':
    train()
