#!/usr/bin/python
# -*- coding: utf-8 -*-

shorten = {
    '<http://dbpedia.org/ontology/' : 'dbo:',
    '<http://dbpedia.org/datatype/' : 'dbot:',
    '<http://dbpedia.org/property/' : 'dbp:',
    '<http://dbpedia.org/resource/' : 'dbr:'}

def shorten_uri(uri):
    for l, s in shorten.items():
        if l in uri:
            uri = uri.replace(l, s)
            break
    uri = uri.rstrip('>')
    return uri

def get_short_uri(triplet_str):
    uri = triplet_str.split(' ')[0]
    for l, s in shorten.items():
        if l in uri:
            uri = uri.replace(l, s)
            break
    uri = uri.rstrip('>')
    return uri

def get_label(triplet_str):
    assert triplet_str.endswith('"@en .')
    triplet = triplet_str.split(' ')
    label = ' '.join(triplet[2:])[:-6].lstrip('"')
    return label

def get_context(triplet_str):
    assert triplet_str.endswith('" .')
    triplet = triplet_str.split(' ')
    context = ' '.join(triplet[2:])[:-3].lstrip('"').strip()
    return context

