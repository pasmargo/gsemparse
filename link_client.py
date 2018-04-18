'''
Web service that implements a mention linker to arbitrary KBs.
Example call: curl 'http://127.0.0.1:5000/linking/get_best?source=onto_type&mention=angelina'
'''

import argparse
import base64
import pickle
import requests
from flask import Flask, request, jsonify
import json
import logging
import numpy as np
import os
import sys

def ground_mention(mention, source='dbr', nbest=10, port=5000):
    """
    mention might be a list of words that form a single mention,
    or a string with a single word.
    """
    if ' ' in mention:
        mention = mention.split(' ')
    if isinstance(mention, list):
        mention = '&'.join('mention='+m for m in mention)
    else:
        mention = 'mention={0}'.format(mention)
    url = "http://127.0.0.1:5000/linking/get_best?source={0}&{1}&nbest={2}".format(
        source, mention, nbest)
    request_result = requests.get(url)
    result = request_result.json()
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", help="Host name (default: localhost)")
    parser.add_argument("--port", help="Port (default: 5000)", default=5000)
    parser.add_argument("--path", help="Path (default: /linker)")
    parser.add_argument("--nbest", nargs='?', type=int, default=10)
    parser.add_argument("--mention", nargs='+',
        required=True,
        help="Mention to ground.")
    parser.add_argument("--source", nargs='?', default="dbr",
        choices=['onto_type', 'onto_rel', 'dbp', 'dbr'])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    host = args.host if args.host else "localhost"
    path = args.path if args.path else "/linking"
    port = int(args.port) if args.port else 5000

    mention = ' '.join(args.mention)
    logging.info('Mention to ground: {0}'.format(mention))
    results = ground_mention(args.mention, args.source, nbest=args.nbest)
    for result in results:
        print(result)

