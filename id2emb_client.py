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

def embed_id(kb_id):
    """
    mention might be a list of words that form a single mention,
    or a string with a single word.
    """
    url = "http://127.0.0.1:5010/embedding/id2emb?id={0}".format(kb_id)
    request_result = requests.get(url)
    result = request_result.json()
    return result

# print(embed_id('dbo:Island'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", help="Knowledge base identifier")
    parser.add_argument("--path", help="Path (default: /embedding)")
    parser.add_argument("--host", help="Host (default: localhost)")
    parser.add_argument("--port", help="Port (default: 5010)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    host = args.host if args.host else "localhost"
    path = args.path if args.path else "/embedding"
    port = int(args.port) if args.port else 5010

    result = embed_id(args.id)
    print(result)

