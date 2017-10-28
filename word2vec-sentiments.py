# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

# random
import random

# numpy
import numpy

# classifier
from sklearn.linear_model import LogisticRegression

import logging
import sys


log = logging.getLogger()
log.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
log.addHandler(handler)


class TaggedLineDocs(object):
    def __init__(self, sources):
        self.sources = sources
        self.sentences = []

        # make sure that prefixes are unique - why do we need this??
        prefixes = set()
        for value in self.sources.values():
            if value not in prefixes:
                prefixes.add(value)
            else:
                raise Exception('Non-unique prefix encountered')

        # load all in memory
        for source, prefix in self.sources.items():
            with open(source, encoding="utf8") as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))

    def __iter__(self):
        return iter(self.sentences)

    def shuffle(self):
        random.shuffle(self.sentences)
        return self


# Start loading dataset and load/train from scratch a model
log.info('source load')
sources = {'test-neg.txt': 'TEST_NEG',
           'test-pos.txt': 'TEST_POS',
           'train-neg.txt': 'TRAIN_NEG',
           'train-pos.txt': 'TRAIN_POS',
           'train-unsup.txt': 'TRAIN_UNS'}
documents = TaggedLineDocs(sources)

log.info('Initializing D2V model')
epochs = 20
vec_size = 400

# model = Doc2Vec(min_count=3, window=10, size=vec_size, sample=1e-4, negative=5, workers=4, dm=0, iter=epochs)

# log.info('Pre-trained D2V Model Load')
model = Doc2Vec.load('./imdb_new.d2v')

# model.build_vocab(documents)

# log.info('Training Epochs %i', epochs)
# model.train(documents, total_examples=model.corpus_count, epochs=model.iter)

# log.info('Model Save')
# model.save('./imdb_new.d2v')


# Classifying sentiment
log.info('Sentiment')
train_arrays = numpy.zeros((25000, vec_size))
train_labels = numpy.zeros(25000)

for i in range(12500):
    prefix_train_pos = 'TRAIN_POS_' + str(i)
    prefix_train_neg = 'TRAIN_NEG_' + str(i)
    train_arrays[i] = model.docvecs[prefix_train_pos]
    train_arrays[12500 + i] = model.docvecs[prefix_train_neg]
    train_labels[i] = 1
    train_labels[12500 + i] = 0

log.info(train_labels)

test_arrays = numpy.zeros((25000, vec_size))
test_labels = numpy.zeros(25000)

for i in range(12500):
    prefix_test_pos = 'TEST_POS_' + str(i)
    prefix_test_neg = 'TEST_NEG_' + str(i)
    test_arrays[i] = model.docvecs[prefix_test_pos]
    test_arrays[12500 + i] = model.docvecs[prefix_test_neg]
    test_labels[i] = 1
    test_labels[12500 + i] = 0

log.info('Fitting')
classifier = LogisticRegression()
classifier.fit(train_arrays, train_labels)

# LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
#                    intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)

log.info(classifier.score(test_arrays, test_labels))
