# TODO:

* '0' as padding might be dominating. At the moment I created a zero embedding for this entry.
  * You can continue testing with python test2.py
* DB encoding results in a suspiciously regular matrix. Investigate that.
* Check why the dbr linking of "angelina jolie" works so bad.
  * When printing the encoding of that mention, I find very small floating values. Possibly due to:
  * A strong regularization,
  * The activation function (tanh),
  * A flawed loss function.
* When the pipeline is completed, return to the entity linker for further refinements. E.g.:
  * Create an evaluation method for the linker. For example, use all keywords from SQA
    and all URIs, and measure the coverage@10 from all keywords.
  * Siamese network for similarity measurements using "Learning to Rank" techniques.
* Write grammar for SQA challenge according to (their templates)[https://github.com/AskNowQA/LC-QuAD/blob/develop/templates.py]:
  * It needs to be refined to include queries of the type `?x0 rdf:type Class`.
* Improve sampling for linking routines:
  * We can create meaningful data augmentation by corrupting
    labels in a natural way (e.g. misspellings, shortening, cropping, etc.).

# Instructions and examples:

## Preprocessing
```
$ python ontology_prep.py dbpedia_ontology.nt > dbpedia_ontology.labels.jsonl
```

Outputs (~4K items):

```
{"role": "type", "uri": "dbo:LunarCrater", "label": "lunar crater"}
{"role": "type", "uri": "dbo:MotorsportSeason", "label": "motorsport season"}
...
{"role": "rel", "uri": "dbo:fansgroup", "label": "fansgroup"}
{"role": "rel", "uri": "dbo:population", "label": "population"}
```

```
$ python dbpedia_rels_prep.py infobox_property_definitions_en.ttl.bz2 > dbpedia_rels.labels.jsonl
```

Otputs (~60K items):

```
{"uri": "dbp:mostSuccessfulRiders", "label": "most successful riders"}
{"uri": "dbp:dateOfBuilt", "label": "date of built"}
...
```

```
$ python dbpedia_ents_prep.py infobox_properties_en.ttl.bz2 labels_en.ttl.bz2 nif_context_en.ttl.bz2 > dbpedia_ents.text.jsonl
```

Outputs (~13M items):

```
{"context": "Rarohenga is the underworld and realm of the spirits in M\u0101ori mythology. Inhabitants of Rarohenga are called turehu and are governed by Hine nui te Po.", "label": "Rarohenga", "surf": "Rarohenga", "uri": "dbr:Rarohenga"}
{"context": "", "label": "Saryesik Atyrau Desert", "surf": "Saryesik_Atyrau_Desert", "uri": "dbr:Saryesik_Atyrau_Desert"}
...
```


## Train linker:

```
CUDA_VISIBLE_DEVICES=6 python linking.py --loss_type i1i1-i1i2s
```

## Linking
To compute entity linking:

```
$ echo "angelina jolie" | CUDA_VISIBLE_DEVICES=7 python el.py --model char-cnn_linkent_i1i1-i1i2s.check
```

Output:

```
{"label": "Angelina jolie", "uri": "dbr:Angelina_jolie", "score": 1.0}
{"label": "Angelina Jolie", "uri": "dbr:Angelina_Jolie", "score": 1.0}
{"label": "Angelina Jolie Trapdoor Spider", "uri": "dbr:Angelina_Jolie_Trapdoor_Spider", "score": 0.9935481548309326}
{"label": "Angelina Jolie Voight", "uri": "dbr:Angelina_Jolie_Voight", "score": 0.9934994578361511}
{"label": "Angelina Jolie cancer treatment", "uri": "dbr:Angelina_Jolie_cancer_treatment", "score": 0.9927096366882324}
{"label": "Angelina Jolie Pitt", "uri": "dbr:Angelina_Jolie_Pitt", "score": 0.9916501045227051}
{"label": "Angelina Joli", "uri": "dbr:Angelina_Joli", "score": 0.9904723167419434}
{"label": "Angelina Jolie Filmography", "uri": "dbr:Angelina_Jolie_Filmography", "score": 0.9891354441642761}
{"label": "Angelina Jolie filmography", "uri": "dbr:Angelina_Jolie_filmography", "score": 0.9891354441642761}
{"label": "Anjelina Jolie", "uri": "dbr:Anjelina_Jolie", "score": 0.9852608442306519}
```

Another example, with misspellings:

```
$ echo "angeline yoli" | CUDA_VISIBLE_DEVICES=7 python -i el.py --model char-cnn_linkent_i1i1-i1i2s.check
```

Output:

```
{"label": "Angeline Jolie", "uri": "dbr:Angeline_Jolie", "score": 0.9747606515884399}
{"label": "Uncle Willie", "uri": "dbr:Uncle_Willie", "score": 0.9626715183258057}
{"label": "Parmelia (lichen)", "uri": "dbr:Parmelia_(lichen)", "score": 0.9624680280685425}
{"label": "Uriele Vitolo", "uri": "dbr:Uriele_Vitolo", "score": 0.9623000025749207}
{"label": "Ding Lieyun", "uri": "dbr:Ding_Lieyun", "score": 0.9620264172554016}
{"label": "Earl of Loudon", "uri": "dbr:Earl_of_Loudon", "score": 0.9615837335586548}
{"label": "Angeline Myra Keen", "uri": "dbr:Angeline_Myra_Keen", "score": 0.9613763093948364}
{"label": "Angel Negro", "uri": "dbr:Angel_Negro", "score": 0.9613314867019653}
{"label": "Angeline Malik", "uri": "dbr:Angeline_Malik", "score": 0.9612295627593994}
{"label": "Angeline of Marsciano", "uri": "dbr:Angeline_of_Marsciano", "score": 0.9602781534194946}
```

## Linking Server API

```
CUDA_VISIBLE_DEVICES=5 python link_api.py --tgs onto_type data/dbpedia_ontology_types.labels.jsonl --tgs onto_rel data/dbpedia_ontology_rels.labels.jsonl --tgs dbp data/dbpedia_rels.labels.jsonl --tgs dbr data/dbpedia_ents.text.jsonl
```

It will start in server mode, listening on port 5000. It may take time to get ready since it needs to load all entities, relations and types from disk.

When the server is ready, you will see a message:

```
INFO:werkzeug: * Running on http://localhost:5000/ (Press CTRL+C to quit)
```

Then, you can use the linking client to do the linking (grounding):

```
python link_client.py --mention sibling --source onto_rel
python link_client.py --mention mountain --source onto_type
```

## Embedding KB identifiers into vectors

First, we need to start the server:

```
python id2emb_api.py --targets data/dbpedia_ontology_types.labels.jsonl
```

This will start the server, loading the jsonl files with URIs and labels, and GoogleNews vector (can be changed).

You can obtain 300-dim vectors for KB identifiers (URIs) doing:

```
python id2emb_client.py --id dbo:Island
```

(if the URI doesn't exist, then the vector is all zeros).

# (old) Review of resources that might be useful to train a linker.

* In DBpedia ontology there are ontological types and relations (30K in total).
* Model: map mentions to types or dbo-relations:
  * Data: dbpedia_ontology.nt, ~ 30K items.
  * Includes labels and comments/descriptions.
* Model: map mentions to dpedia resources (dbr).
  * Data: infobox_properties_en.ttl.bz2, ~3M unique items.
  * Includes: dbpedia.org/resource/<Name> where <Name> could be used as a large-coverage character match.
  * Data: labels_en.ttl.bz2, ~12M unique (?) items.
  * Includes: df-schema#label short English strings.
  * Data: nif_context_en.ttl.bz2, ~5M items.
  * Includes: isString large context (one big paragraph).
* Model: map mentions to dpedia properties (dbp).
  * Data: infobox_property_definitions_en.ttl.bz2, 60K items.
  * Includes: rdf-schema#label with a meaningful short English label.

