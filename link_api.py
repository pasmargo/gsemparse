'''
Simple web service wrapping a Word2Vec as implemented in Gensim
Example call: curl http://127.0.0.1:5000/wor2vec/n_similarity/ws1=Sushi&ws1=Shop&ws2=Japanese&ws2=Restaurant
@TODO: Add more methods
@TODO: Add command line parameter: path to the trained model
@TODO: Add command line parameters: host and port
'''

import argparse
import base64
import pickle
from flask import Flask, request, jsonify
from flask_restful import Resource, Api, reqparse
import json
import logging
import numpy as np
from numpy import exp, dot, zeros, outer, random, dtype, get_include, float32 as REAL,\
     uint32, seterr, array, uint8, vstack, argsort, fromstring, sqrt, newaxis, ndarray, empty, sum as np_sum
import os
import sys

from keras.models import Model
import sklearn.metrics.pairwise as pairwise

from models import make_encoder
from preprocessing import labels_to_matrix
from preprocessing import load_labels

parser = reqparse.RequestParser()

# class Similarity(Resource):
#     def get(self):
#         parser = reqparse.RequestParser()
#         parser.add_argument('w1', type=str, required=True, help="Word 1 cannot be blank!")
#         parser.add_argument('w2', type=str, required=True, help="Word 2 cannot be blank!")
#         args = parser.parse_args()
#         return model.similarity(args['w1'], args['w2'])
# 
# class ModelWordSet(Resource):
#     def get(self):
#         try:
#             res = base64.b64encode(cPickle.dumps(set(model.index2word)))
#             return res
#         except Exception, e:
#             print e
#             return

class GetBest(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('source', type=str, default='dbr',
            help="Source where to search (e.g. Ontology types, Ontology relations, DPBedia relations, etc.)",
            choices=['onto_type', 'onto_rel', 'dbp', 'dbr'])
        parser.add_argument('mention', type=str, required=True, action='append',
            help="Mention from the claim or question")
        parser.add_argument('nbest', type=int, required=False, default=10,
            help="Number of results.")
        parser.add_argument('fields', type=str, default='uri,label',
            help="Fields to return as the URI information",
            choices=['onto_type', 'onto_rel', 'dbp', 'dbr'])
        args = parser.parse_args()
        mention = ' '.join(args.mention).lower()
        M = labels_to_matrix([mention])
        M_enc = encoder.predict(M)
        try:
            L_enc = vectors_by_source[args.source]
            uri_infos = uri_infos_by_source[args.source]
        except Exception as e:
            print(e)
            return

        logging.info('Computing pairwise similarities for mention {0} in {1}.'.format(
            mention, source))
        diffs = pairwise.pairwise_distances(M_enc, X_enc, metric='cosine', n_jobs=-1)
        logging.info('Finished computing similarities.')

        fields = args.fields.split(',')

        diffs_argpart = np.argpartition(diffs, args.nbest)
        best_uris = []
        for i in best_entries:
            uri_info = entities[i]
            for field in list(uri_info.keys()):
                if field not in fields:
                    del uri_info[field]
            uri_info['score'] = 1 - diffs[0][i]
            best_uris.append(uri_info)
        logging.info('Mention: {0}, best URIs: {1}'.format(mention, best_uris))
        best_uris.sort(key=lambda e: e['score'], reverse=True)
        result = base64.b64encode(pickle.dumps(best_uris))
        return result

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
        default="char-cnn_linkent_i1i1-i1i2s.check",
        help="Filename where the Keras encoder is saved.")
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
    inputs, outputs, char_emb_x = make_encoder()
    encoder = Model(inputs=inputs, outputs=outputs)
    encoder.load_weights(args.model, by_name=True)

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

    api.add_resource(Linking, path+'/get_best')
    app.run(host=host, port=port)


