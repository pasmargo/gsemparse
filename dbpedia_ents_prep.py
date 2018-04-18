#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
import bz2
import codecs
from collections import defaultdict
import json
import logging
import sys

from ttl_prep import get_context
from ttl_prep import get_label
from ttl_prep import get_short_uri
from ttl_prep import shorten_uri

logging.basicConfig(level=logging.INFO)

infobox_fname = sys.argv[1]
labels_fname = sys.argv[2]
context_fname = sys.argv[3]

uris = set()
with bz2.BZ2File(infobox_fname, 'r') as fin_info:
    for triplet_str in fin_info:
        triplet_str = triplet_str.decode('utf-8')
        triplet_str = triplet_str.strip().split()[0]
        uris.add(triplet_str)
uris = set(shorten_uri(uri) for uri in uris)
logging.info('Read {0} unique uris from {1}'.format(len(uris), infobox_fname))

info = defaultdict(lambda: {'uri': '', 'surf': '', 'label': '', 'context': ''})

with bz2.BZ2File(labels_fname, 'r') as fin_labels:
    for triplet_str in fin_labels:
        triplet_str = triplet_str.decode('utf-8') # cast byte type data to string
        triplet_str = triplet_str.strip()
        if 'rdf-schema#label' in triplet_str and triplet_str.endswith('"@en .'):
            uri = get_short_uri(triplet_str)
            label = get_label(triplet_str)
            info[uri]['label'] = label

with bz2.BZ2File(context_fname, 'r') as fin_context:
    for triplet_str in fin_context:
        triplet_str = triplet_str.decode('utf-8') # cast byte type data to string
        triplet_str = triplet_str.strip()
        if 'nif-core#isString' in triplet_str and triplet_str.endswith('" .'):
            uri_long = triplet_str.split()[0][:-26]
            uri = shorten_uri(uri_long)
            context = get_context(triplet_str)
            info[uri]['context'] = context

for uri, values in info.items():
    values['uri'] = uri
    values['surf'] = ':'.join(uri.split(':')[1:])
    print(json.dumps(values, sort_keys=True, indent=None))

logging.info('Produced {0} entity entries (with varying amount of textual information)'.format(
    len(info)))
