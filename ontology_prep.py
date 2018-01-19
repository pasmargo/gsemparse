#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
import codecs
import json
import logging
import sys

from ttl_prep import get_label
from ttl_prep import get_short_uri

ontology_fname = sys.argv[1]
dbo = {}
# Store whether the ontology entry is a type (class) or a relation.
dbo_role = {}

with codecs.open(ontology_fname, 'r', 'utf-8') as fin:
    for triplet_str in fin:
        triplet_str = triplet_str.strip()
        if 'syntax-ns#type' in triplet_str:
            uri = get_short_uri(triplet_str)
            if 'Property>' in triplet_str:
                dbo_role[uri] = 'rel'
            elif 'owl#Class' in triplet_str or 'rdf-schema#Datatype' in triplet_str:
                dbo_role[uri] = 'type'
            else:
                logging.warning('Ontology entry not recognized: {0}'.format(triplet_str))
        if 'rdf-schema#label' in triplet_str and triplet_str.endswith('"@en .'):
            uri = get_short_uri(triplet_str)
            label = get_label(triplet_str)
            role = dbo_role.get(uri, None)
            assert role is not None
            dbo[uri] = {'uri' : uri, 'label' : label, 'role' : role}
        if 'rdf-schema#comment' in triplet_str and triplet_str.endswith('"@en .'):
            uri = get_short_uri(triplet_str)
            comment = get_label(triplet_str)
            if uri not in dbo:
                logging.warning('Comment in {0} for URI not recognized'.format(triplet_str))
            else:
                dbo[uri]['context'] = comment

for item in dbo.values():
    if 'context' not in item:
        item['context'] = ""
    print(json.dumps(item, indent=None))
