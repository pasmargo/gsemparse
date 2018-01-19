
import codecs
import json
import logging
import numpy as np
import re
import string
import time

chars = string.ascii_lowercase + string.digits + string.punctuation
char_indices = dict((c, i) for i, c in enumerate(chars))

logging.basicConfig(level=logging.INFO)

def time_count(fn):
  def _wrapper(*args, **kwargs):
    start = time.clock()
    returns = fn(*args, **kwargs)
    print("[time_count]: %s took %fs" % (fn.__name__, time.clock() - start))
    return returns
  return _wrapper

def binarize(x, sz=71):
    return tf.to_float(tf.one_hot(x, sz, on_value=1, off_value=0, axis=-1))

def binarize_outshape(in_shape):
    return in_shape[0], in_shape[1], 71

def striphtml(s):
    p = re.compile(r'<.*?>')
    return p.sub('', s)

def clean(s):
    return re.sub(r'[^\x00-\x7f]', r'', s)

@time_count
def load_labels(fname, ntrain=1000, lowercase=True, return_jsonl=False):
    labels = []
    num_ignored = 0
    num_labels = 0
    jsonl_data = []
    with codecs.open(fname, 'r', 'utf-8') as fin:
        for json_line in fin:
            d = json.loads(json_line.strip())
            label = d.get('label', '')
            if len(label) > 0:
                labels.append(label.lower())
                num_labels += 1
                if return_jsonl:
                    jsonl_data.append(d)
            else:
                num_ignored += 1
            if num_labels == ntrain:
                break
    logging.info('Loaded {0} labels, {1} ignored, {2} requested.'.format(
        len(labels), num_ignored, ntrain))
    if return_jsonl:
        return labels, jsonl_data
    return labels

def labels_to_matrix(labels, maxlen=16):
    X = np.ones((len(labels), maxlen), dtype=np.int32) * -1
    for i, label in enumerate(labels):
        for j, char in enumerate(label[:maxlen]):
            X[i, (maxlen - 1 - j)] = char_indices.get(char, -1)
    return X

