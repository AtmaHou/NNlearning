import math
import os
import random
import sys
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from  data_utils import *
from  seq2seq_model import *
import codecs

_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

def read_chat_data(data_path,vocabulary_path, max_size=None):
    counter = 0
    vocab, _ = initialize_vocabulary(vocabulary_path)
    print(len(vocab))
    print(max_size)
    data_set = [[] for _ in _buckets]
    # http://stackoverflow.com/questions/33054527/python-3-5-typeerror-a-bytes-like-object-is-required-not-str-when-writing-t
    with codecs.open(data_path, "rb") as fi:
        for line in fi.readlines():
            line = line.decode('utf8').strip()
            counter += 1
            if max_size!=0 and counter > max_size:
                break
            if counter % 10000 == 0:
              print("  reading data line %d" % counter)
              sys.stdout.flush()
            entities = line.lower().split("|")
            # print entities
            if len(entities) == 2:
                source = entities[0]
                target = entities[1]
                source_ids = [int(x) for x in sentence_to_token_ids(source,vocab)]
                target_ids = [int(x) for x in sentence_to_token_ids(target,vocab)]
                target_ids.append(EOS_ID)
                for bucket_id, (source_size, target_size) in enumerate(_buckets):
                  if len(source_ids) < source_size and len(target_ids) < target_size:
                    data_set[bucket_id].append([source_ids, target_ids])
                    break
    return data_set


data_path = 'ubuntu/my_train_2'#'ubuntu/my_train_1.tsv'
dev_data = 'ubuntu/valid.tsv'
vocab_path = 'ubuntu/my_train_2_vocab'#'ubuntu/60k_vocan.en'
train_set = read_chat_data(data_path,vocab_path, 0)
train_set
