'''
Web service that implements a mention linker to arbitrary KBs.
Example call: curl 'http://127.0.0.1:5000/linking/get_best?source=onto_type&mention=angelina'
'''

import argparse
import base64
import pickle
from flask import Flask, request, jsonify
import json
import logging
import numpy as np
import os
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", help="Host name (default: localhost)")
    parser.add_argument("--port", help="Port (default: 5000)")
    parser.add_argument("--path", help="Path (default: /linker)")
    parser.add_argument("--targets", nargs='?',
        default="data/dbpedia_ents.text.jsonl",
        help="Filename where the KB items (uri, label) are in jsonl.")
    parser.add_argument('--tgs', action='append', nargs=2,
        metavar=('source', 'filename'),
        help='Tuple with source (e.g. ontology type, DBpedia resource, etc.) and filename with jsonl entries')
    parser.add_argument("--nbest", nargs='?', type=int, default=10)
    parser.add_argument("--model", nargs='?',
        default="char-cnn_linkent_i1i1-i1i2s.h5",
        help="Filename where the Keras encoder model is saved.")
    parser.add_argument("--model_type", nargs='?', default="char-cnn",
        choices=["char-cnn", "char-lstm"])
    parser.add_argument("--nlabels", nargs='?', type=int, default=-1,
        help="Number of entities to load from file. If -1, then load them all.")
    parser.add_argument("--maxlen", nargs='?', type=int, default=16,
        help="Maximum length of labels. Longer labels are cropped.")
    parser.add_argument("--fields", nargs='?', default="uri,label")
    parser.add_argument("--char_emb_size", nargs='?', type=int, default=128)
    parser.add_argument("--batch_size", nargs='?', type=int, default=1000)
    parser.set_defaults(reduce_params=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    model_path = args.model if args.model else "./model.bin.gz"
    host = args.host if args.host else "localhost"
    path = args.path if args.path else "/linking"
    port = int(args.port) if args.port else 5000

    

