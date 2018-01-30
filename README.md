# TODO:
* When the pipeline is completed, return to the entity linker for further refinements. E.g.:
  * Siamese network for similarity measurements using "Learning to Rank" techniques.
* Improve sampling:
  * We can create meaningful data augmentation by corrupting
    labels in a natural way (e.g. misspellings, shortening, cropping, etc.).

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

