'''
Web service that implements a mention linker to arbitrary KBs.
Example call: curl 'http://127.0.0.1:5000/embedding/id2emb?id=dbo:sibling'
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

import gensim

from preprocessing import load_labels

rparser = reqparse.RequestParser()
rparser.add_argument('id', type=str, required=True,
    help=("Knowledge base identifier (entity, relation or class) for which"
          " the embedding is desired. This is case sensitive."))
rparser.add_argument('fields', type=str, default='uri,label,role',
    help="Fields to return as the URI information")

kEmbedDim = 300

class Embed(Resource):
    def get(self):
        global rparser
        args = rparser.parse_args()
        kb_id = args['id']
        uri_info = uri_to_info.get(kb_id, {})
        result = np.zeros((kEmbedDim,), dtype='float32')
        if uri_info and 'label' in uri_info:
            label = uri_info.get('label', '')
            words = [word for word in label.split(' ') if word in word2vec]
            print('Queried: {0}. Label: {1}. Words in model: {2}'.format(
                kb_id, label, words))
            if words:
                for word in words:
                    result += word2vec[word]
                result /= len(words)
            print('Result: {0}'.format(result))
            # return base64.b64encode(result)
        result_list = result.tolist()
        return jsonify(result_list)

# fields = args['fields'].split(',')
# for field in list(uri_info.keys()):
#     if field not in fields:
#         del uri_info[field]
# return uri_info

app = Flask(__name__)
api = Api(app)

@app.errorhandler(404)
def pageNotFound(error):
    return "page not found"

@app.errorhandler(500)
def raiseError(error):
    return error

if __name__ == '__main__':
    global uri_to_info
    global word2vec

    #----------- Parsing Arguments ---------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", help="Host name (default: localhost)")
    parser.add_argument("--port", help="Port (default: 5010)")
    parser.add_argument("--path", help="Path (default: /embedding)")
    parser.add_argument("--word2vec_model", nargs='?', type=str,
        default="GoogleNews-vectors-negative300.bin")
    parser.add_argument("--targets", nargs='+',
        default="data/dbpedia_ents.text.jsonl",
        help="Filename where the KB items (uri, label) are in jsonl.")
    parser.add_argument("--nlabels", nargs='?', type=int, default=-1,
        help="Number of entities to load from file. If -1, then load them all.")
    parser.add_argument("--fields", nargs='?', default="uri,label")
    parser.set_defaults(reduce_params=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    host = args.host if args.host else "localhost"
    path = args.path if args.path else "/embedding"
    port = int(args.port) if args.port else 5010

    uri_to_info = {}
    for ifname in args.targets:
        if not os.path.exists(ifname):
            print('Target file (with labels) does not exist: {0}'.format(ifname))
            parser.print_help(file=sys.stderr)
            sys.exit(1)
        else:
            logging.info('Loading targets in {0} ...'.format(ifname))
            labels, uri_infos = load_labels(
                ifname,
                ntrain=args.nlabels,
                return_jsonl=True)
            assert len(labels) == len(uri_infos)
            uri_to_info.update({d['uri'] : d for d in uri_infos})
    print('Some items read: {0}'.format(list(uri_to_info.items())[:10]))
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(
        args.word2vec_model, binary=True)

    api.add_resource(Embed, path + '/id2emb')
    app.run(host=host, port=port)


