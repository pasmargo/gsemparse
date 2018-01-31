'''
Web service that implements a mention linker to arbitrary KBs.
Example call: curl 'http://127.0.0.1:5000/linking/get_best?source=onto_type&mention=angelina'
'''

import argparse
import base64
import pickle
from flask import Flask, request, jsonify
from flask_restful import Resource, Api, reqparse
import json
import logging
import numpy as np
import os
import sys

from keras.models import load_model
import sklearn.metrics.pairwise as pairwise

from models import make_encoder
from preprocessing import labels_to_matrix
from preprocessing import load_labels

rparser = reqparse.RequestParser()
rparser.add_argument('source', type=str, default='dbr',
    help="Source where to search (e.g. Ontology types, Ontology relations, DPBedia relations, etc.)",
    choices=['onto_type', 'onto_rel', 'dbp', 'dbr'])
rparser.add_argument('mention', type=str, required=True, action='append',
    help="Mention from the claim or question")
rparser.add_argument('nbest', type=int, required=False, default=10,
    help="Number of results.")
rparser.add_argument('fields', type=str, default='uri,label,role',
    help="Fields to return as the URI information",
    choices=['onto_type', 'onto_rel', 'dbp', 'dbr'])

class GetBest(Resource):
    def get(self):
        global rparser
        args = rparser.parse_args()
        mention = ' '.join(args['mention']).lower()
        source = args['source']
        M = labels_to_matrix([mention])
        M_enc = encoder.predict(M)
        print('Encoding of {0}:\n{1}'.format(mention, M_enc))
        try:
            L_enc = vectors_by_source[source]
            uri_infos = uri_infos_by_source[source]
        except Exception as e:
            print(e)
            print('Source {0} not available. Try instead one of: {1}'.format(
                source, list(vectors_by_source.keys())))
            raise

        logging.info('Computing pairwise similarities for mention "{0}" as {1}.'.format(
            mention, source))
        diffs = pairwise.pairwise_distances(M_enc, L_enc, metric='cosine', n_jobs=-1)
        logging.info('Finished computing similarities.')

        fields = args['fields'].split(',')

        diffs_argpart = np.argpartition(diffs, args['nbest'])
        best_entries = list(diffs_argpart[0][:args['nbest']])
        best_uris = []
        for i in best_entries:
            try:
                uri_info = uri_infos[i]
            except Exception as e:
                print(e)
                raise
            for field in list(uri_info.keys()):
                if field not in fields:
                    del uri_info[field]
            uri_info['score'] = 1 - diffs[0][i]
            best_uris.append(uri_info)
        best_uris.sort(key=lambda e: e['score'], reverse=True)
        logging.info('Mention: "{0}", best URIs: {1}'.format(mention, best_uris))
        return best_uris

app = Flask(__name__)
api = Api(app)

@app.errorhandler(404)
def pageNotFound(error):
    return "page not found"

@app.errorhandler(500)
def raiseError(error):
    return error

if __name__ == '__main__':
    global model
    global encoder
    global labels_by_source
    global uri_infos_by_source
    global matrix_by_source
    global vectors_by_source

    #----------- Parsing Arguments ---------------
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

    if not os.path.exists(args.model):
        print('Model file does not exist: {0}'.format(args.model))
        parser.print_help(file=sys.stderr)
        sys.exit(1)
    encoder = load_model(args.model)

    labels_by_source = {}
    uri_infos_by_source = {}
    matrix_by_source = {}
    vectors_by_source = {}
    for source, ifname in args.tgs:
        if not os.path.exists(ifname):
            print('Target file (with labels does not exist: {0}'.format(ifname))
            parser.print_help(file=sys.stderr)
            sys.exit(1)
        else:
            logging.info('Loading targets in {0} ...'.format(ifname))
            labels, uri_infos = load_labels(
                ifname,
                ntrain=args.nlabels,
                return_jsonl=True)
            assert len(labels) == len(uri_infos)
            X = labels_to_matrix(labels, args.maxlen)
            logging.info('Source {0}, matrix {1}'.format(source, X.shape))
            labels_by_source[source] = labels
            uri_infos_by_source[source] = uri_infos
            matrix_by_source[source] = X
            X_enc = encoder.predict(X, batch_size=args.batch_size)
            vectors_by_source[source] = X_enc

    api.add_resource(GetBest, path + '/get_best')
    app.run(host=host, port=port)


