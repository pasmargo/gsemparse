import argparse
import json
import logging
import os
import sys

from keras.models import Model
import numpy as np
import sklearn.metrics.pairwise as pairwise

from models import make_encoder
from models import make_decoder
from models import make_label_input
from models import make_siamese_model
from preprocessing import char_indices
from preprocessing import labels_to_matrix
from preprocessing import load_labels

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "queries", nargs='?', type=argparse.FileType('r'), default=sys.stdin)
    parser.add_argument("--mentions", nargs='?', default="")
    parser.add_argument("--nbest", nargs='?', type=int, default=10)
    parser.add_argument("--model", nargs='?',
        default="char-cnn_linkent_i1i1-i1i2s.check",
        help="Filename where the Keras encoder is saved.")
    parser.add_argument("--model_type", nargs='?', default="char-cnn",
        choices=["char-cnn", "char-lstm"])
    parser.add_argument("--loss_type", nargs='?', default="i1i1",
        choices=["i1i1", "i1i1-i1i2s", "i1i1-i1i2s-i1j1s"])
    parser.add_argument("--nlabels", nargs='?', type=int, default=-1,
        help="Number of entities to load from file. If -1, then load them all.")
    parser.add_argument("--maxlen", nargs='?', type=int, default=16,
        help="Maximum length of labels. Longer labels are cropped.")
    parser.add_argument("--fields", nargs='?', default="uri,label")
    parser.add_argument("--char_emb_size", nargs='?', type=int, default=128)
    parser.add_argument("--batch_size", nargs='?', type=int, default=500)
    parser.add_argument("--batch_size_val", nargs='?', type=int, default=150)
    parser.add_argument("--epochs", nargs='?', type=int, default=200)
    parser.add_argument("--steps_per_epoch", nargs='?', type=int, default=100)
    parser.add_argument("--validation_steps", nargs='?', type=int, default=1)
    parser.add_argument("--patience", nargs='?', type=int, default=5)
    parser.add_argument("--batch_normalization", nargs='?', type=bool, default=False)
    parser.add_argument("--reduce_params", dest="reduce_params", action="store_true")
    parser.add_argument("--no_reduce_params", dest="reduce_params", action="store_false")
    parser.add_argument("--exp_suffix", nargs='?', default="")
    parser.set_defaults(reduce_params=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    model_fname = args.model
    if not os.path.exists(model_fname):
        print('Model file does not exist: {0}'.format(model_fname))
        parser.print_help(file=sys.stderr)
        sys.exit(1)

    labels, entities = load_labels(
        'data/dbpedia_ents.text.jsonl',
        ntrain=args.nlabels,
        return_jsonl=True)
    assert len(labels) == len(entities)
    X = labels_to_matrix(labels, args.maxlen)

    mentions = [m.strip().lower() for m in args.queries]
    M = labels_to_matrix(mentions)

    num_filters = (args.char_emb_size, args.char_emb_size * 2, args.char_emb_size * 4)
    filter_lengths = (3, 3, 3)
    subsamples = (1, 1, 1)
    pool_lengths = (2, 2, 2)

    inputs, outputs, char_emb_x = make_encoder(
        args.maxlen,
        args.char_emb_size,
        num_filters=num_filters,
        filter_lengths=filter_lengths,
        subsamples=subsamples,
        pool_lengths=pool_lengths)
    encoder = Model(inputs=inputs, outputs=outputs)
    encoder.load_weights(model_fname, by_name=True)

    ent_cache_fname = model_fname.replace('.check', '_ent{0}_enc.npy'.format(args.nlabels))
    if os.path.exists(ent_cache_fname):
        logging.info('Loading cached embeddings of entities...')
        X_enc = np.load(ent_cache_fname)
        logging.info('Finished loading cached embeddings: {0}'.format(X_enc.shape))
    else:
        logging.info('Computing embeddings of entities...')
        X_enc = encoder.predict(X, batch_size=args.batch_size)
        logging.info('Finished computing embeddings of entities and saving...')
        np.save(ent_cache_fname, X_enc)
        logging.info('Saved to: {0}.'.format(ent_cache_fname))
    M_enc = encoder.predict(M, batch_size=args.batch_size)

    fields = args.fields.split(',')

    logging.info('Computing pairwise similarities between query and KB entities...')
    diffs = pairwise.pairwise_distances(M_enc, X_enc, metric='cosine', n_jobs=-1)
    logging.info('Finished computing similarities.')

    diffs_argpart = np.argpartition(diffs, args.nbest)
    best_entries = list(diffs_argpart[0][:args.nbest])
    for i in best_entries:
        entity = entities[i]
        for field in list(entity.keys()):
            if field not in fields:
                del entity[field]
            entity['score'] = 1 - diffs[0][i]
        print(json.dumps(entity, indent=None))

