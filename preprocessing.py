
import codecs
import json
import logging

import numpy as np
import re
import string

chars = string.ascii_lowercase + string.digits + string.punctuation
char_indices = dict((c, i) for i, c in enumerate(chars))

logging.basicConfig(level=logging.INFO)

def binarize(x, sz=71):
    return tf.to_float(tf.one_hot(x, sz, on_value=1, off_value=0, axis=-1))

def binarize_outshape(in_shape):
    return in_shape[0], in_shape[1], 71

def striphtml(s):
    p = re.compile(r'<.*?>')
    return p.sub('', s)

def clean(s):
    return re.sub(r'[^\x00-\x7f]', r'', s)

def load_labels(fname, ntrain=1000, lowercase=True):
    labels = []
    num_ignored = 0
    num_labels = 0
    with codecs.open(fname, 'r', 'utf-8') as fin:
        for json_line in fin:
            d = json.loads(json_line.strip())
            label = d.get('label', '')
            if len(label) > 0:
                labels.append(label.lower())
                num_labels += 1
            else:
                num_ignored += 1
            if num_labels == ntrain:
                break
    logging.info('Loaded {0} labels, {1} ignored, {2} requested.'.format(
        len(labels), num_ignored, ntrain))
    return labels

def labels_to_matrix(labels, maxlen=16):
    X = np.ones((len(labels), maxlen), dtype=np.int32) * -1
    for i, label in enumerate(labels):
        for j, char in enumerate(label[-maxlen:]):
            X[i, (maxlen - 1 - j)] = char_indices.get(char, -1)
    return X

