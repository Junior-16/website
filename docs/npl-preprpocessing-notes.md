# Natural Language Processing 

## Stop Words
Preprocessing step that aims to remove irrelevant words from
a documents set, like pronoums (I, You, He, This, That, Those, etc),
articles (an, a), prepositions (to, for, from, about, in, etc) and
plurals.

## Tokenization
Main goal is to split a document (aka string) into individual words,
e.g. `"Aoi aoi ano sora"` a document, becomes `["Aoi", "Aoi", "ano", "sora"]`
a sequence of **tokens**.

## Stemming
Process concerned in reduce a word into his root form, for instance 
**get** is the root form of **getting**; **studies**, **studiyng** turns
into **stud**.

## Lemmatization
Process of grouping the inflected form of a word so they can be analysed
as individual item, identified by the word's lemma. In a dictionary structure
words like: **running**, **ran**, and **runned** can referenced by its lemma
wich is **run**.
