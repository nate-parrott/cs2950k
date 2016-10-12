from net import Net, batch_generator
import tensorflow as tf
import numpy as np
from util import rand_variable

NOTHING = "*NOTHING*"


def load_txt(name, ngrams=2):
    grams = []
    for line in open(name):
        words = [NOTHING] * ngrams + line.strip().split()
        for i in xrange(ngrams, len(words)):
            ngram = words[i-ngrams:i]
            word = words[i]
            grams.append((ngram, word))
    return grams

train = load_txt('train.txt')
print len(train) / (20 * 50)
