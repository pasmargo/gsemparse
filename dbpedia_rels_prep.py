#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
import bz2
import codecs
import json
import logging
import sys

from ttl_prep import get_label
from ttl_prep import get_short_uri

ifname = sys.argv[1]

with bz2.BZ2File(ifname, 'r') as fin:
    for triplet_str in fin:
        triplet_str = triplet_str.strip()
        if 'rdf-schema#label' in triplet_str and triplet_str.endswith('"@en .'):
            uri = get_short_uri(triplet_str)
            label = get_label(triplet_str)
            print(json.dumps({'uri' : uri, 'label' : label}, indent=None))

