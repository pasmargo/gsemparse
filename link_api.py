'''
Simple web service wrapping a Word2Vec as implemented in Gensim
Example call: curl http://127.0.0.1:5000/wor2vec/n_similarity/ws1=Sushi&ws1=Shop&ws2=Japanese&ws2=Restaurant
@TODO: Add more methods
@TODO: Add command line parameter: path to the trained model
@TODO: Add command line parameters: host and port
'''

from flask import Flask, request, jsonify
from flask.ext.restful import Resource, Api, reqparse
from gensim.models.word2vec import Word2Vec as w
from gensim import utils, matutils
from numpy import exp, dot, zeros, outer, random, dtype, get_include, float32 as REAL,\
     uint32, seterr, array, uint8, vstack, argsort, fromstring, sqrt, newaxis, ndarray, empty, sum as np_sum
import cPickle
import argparse
import base64
import sys

parser = reqparse.RequestParser()


def filter_words(words):
    if words is None:
        return
    return [word for word in words if word in model.vocab]


class N_Similarity(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('ws1', type=str, required=True, help="Word set 1 cannot be blank!", action='append')
        parser.add_argument('ws2', type=str, required=True, help="Word set 2 cannot be blank!", action='append')
        args = parser.parse_args()
        return model.n_similarity(filter_words(args['ws1']),filter_words(args['ws2']))


class Similarity(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('w1', type=str, required=True, help="Word 1 cannot be blank!")
        parser.add_argument('w2', type=str, required=True, help="Word 2 cannot be blank!")
        args = parser.parse_args()
        return model.similarity(args['w1'], args['w2'])


class MostSimilar(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('positive', type=str, required=False, help="Positive words.", action='append')
        parser.add_argument('negative', type=str, required=False, help="Negative words.", action='append')
        parser.add_argument('topn', type=int, required=False, help="Number of results.")
        args = parser.parse_args()
        pos = filter_words(args.get('positive', []))
        neg = filter_words(args.get('negative', []))
        t = args.get('topn', 10)
        pos = [] if pos == None else pos
        neg = [] if neg == None else neg
        t = 10 if t == None else t
        print "positive: " + str(pos) + " negative: " + str(neg) + " topn: " + str(t)
        try:
            res = model.most_similar_cosmul(positive=pos,negative=neg,topn=t)
            return res
        except Exception, e:
            print e
            print res


class Model(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('word', type=str, required=True, help="word to query.")
        args = parser.parse_args()
        try:
            res = model[args['word']]
            res = base64.b64encode(res)
            return res
        except Exception, e:
            print e
            return

class ModelWordSet(Resource):
    def get(self):
        try:
            res = base64.b64encode(cPickle.dumps(set(model.index2word)))
            return res
        except Exception, e:
            print e
            return

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
    binary = True if args.binary else False
    host = args.host if args.host else "localhost"
    path = args.path if args.path else "/word2vec"
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
    # model = w.load_word2vec_format(model_path, binary=binary)


    api.add_resource(N_Similarity, path+'/n_similarity')
    api.add_resource(Similarity, path+'/similarity')
    api.add_resource(MostSimilar, path+'/most_similar')
    api.add_resource(Model, path+'/model')
    api.add_resource(ModelWordSet, '/word2vec/model_word_set')
    app.run(host=host, port=port)


mentions = [m.strip().lower() for m in args.queries]
# mentions = ['colour']
M = labels_to_matrix(mentions)

M_enc = encoder.predict(M, batch_size=args.batch_size)

fields = args.fields.split(',')

logging.info('Computing pairwise similarities between query and KB entities...')
diffs = pairwise.pairwise_distances(M_enc, X_enc, metric='cosine', n_jobs=-1)
logging.info('Finished computing similarities.')

diffs_argpart = np.argpartition(diffs, args.nbest)
best_entries = list(diffs_argpart[0][:args.nbest])
print(best_entries)
best_entities = []
# from pudb import set_trace; set_trace()
for i in best_entries:
    entity = entities[i]
    for field in list(entity.keys()):
        if field not in fields:
            del entity[field]
    entity['score'] = 1 - diffs[0][i]
    best_entities.append(entity)
print(best_entities)
best_entities.sort(key=lambda e: e['score'], reverse=True)
for entity in best_entities:
    print(json.dumps(entity, indent=None))

