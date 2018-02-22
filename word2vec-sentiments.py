# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

# random
import random
import numpy.random as nprandom

# numpy
import numpy

# classifier
from sklearn.linear_model import LogisticRegression

import logging
import sys
import os


log = logging.getLogger('Doc2VecSentiments')
log.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter('%(asctime)s %(name)s-%(levelname)s %(message)s'))
log.addHandler(handler)

# constant seed for reproducible results
seed = 12347
random.seed(seed)
nprandom.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)


class TaggedLineDocs(object):
    def __init__(self, sources):
        self.sentences = []

        # load all in memory TODO make back again streaming from files in iter
        for prefix, source in sources.items():
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
sources = {'TEST_NEG': 'test-neg.txt',
           'TEST_POS': 'test-pos.txt',
           'TRAIN_NEG': 'train-neg.txt',
           'TRAIN_POS': 'train-pos.txt',
           'TRAIN_UNS': 'train-unsup.txt'}
documents = TaggedLineDocs(sources)
log.info('loaded %i documents', len(documents.sentences))


epochs = 30
vec_size = 100

log.info('Initializing D2V model')
model = Doc2Vec(min_count=3, window=10, size=vec_size, sample=1e-4, negative=5, workers=4, dm=0,
                seed=seed,
                iter=epochs)
model.build_vocab(documents)

log.info('Training D2V Epochs %i', epochs)
model.train(documents, total_examples=model.corpus_count, epochs=model.iter)

log.info('Model Save')
model.save('./imdb.d2v')

# log.info('Load Pre-trained D2V Model')
# model = Doc2Vec.load('./imdb.d2v')


# Classifying sentiment
train_arrays = numpy.zeros((25000, vec_size))
train_labels = numpy.zeros(25000)

for i in range(12500):
    prefix_train_pos = 'TRAIN_POS_' + str(i)
    prefix_train_neg = 'TRAIN_NEG_' + str(i)
    train_arrays[i] = model.docvecs[prefix_train_pos]
    train_arrays[12500 + i] = model.docvecs[prefix_train_neg]
    train_labels[i] = 1
    train_labels[12500 + i] = 0

test_arrays = numpy.zeros((25000, vec_size))
test_labels = numpy.zeros(25000)

for i in range(12500):
    prefix_test_pos = 'TEST_POS_' + str(i)
    prefix_test_neg = 'TEST_NEG_' + str(i)
    test_arrays[i] = model.docvecs[prefix_test_pos]
    test_arrays[12500 + i] = model.docvecs[prefix_test_neg]
    test_labels[i] = 1
    test_labels[12500 + i] = 0

# classifier = LogisticRegression()
classifier = LogisticRegression(C=1.5, class_weight=None, dual=False, fit_intercept=True,
                                intercept_scaling=1, penalty='l2', tol=0.0001,
                                random_state=seed,
                                solver='liblinear', max_iter=500)

log.info('Fitting classifier...')
classifier.fit(train_arrays, train_labels)

log.info('Score: {:.3%}'.format(classifier.score(test_arrays, test_labels)))
