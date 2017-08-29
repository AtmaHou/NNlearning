import json
from multiprocessing import Pool, Array, Manager
from functools import partial
import os
import pandas as pd
import re
import itertools
from collections import Counter
import jieba
import pandas as pd
pool_num = 16


base_dir = '/home/data/ljc/film_caption/corpus_ncm/'
caption_file = base_dir + 'srt.out'
vocab_file = base_dir + 'film_vocab_srt'
data_file = base_dir + 'film_data_srt'
# base_dir = '/home/data/ljc/film_caption/corpus/'
# caption_file = base_dir + 'srt.out'
# vocab_file = base_dir + 'film_vocab'
# data_file = base_dir + 'film_data'
# post_file = base_dir + 'post.index'
# response_file = base_dir + 'response.index'
# original_pair_file = base_dir + 'original.pair'
# output_data_file = base_dir + 'noah_data'

# with open(base_dir+'film_vocab_most', 'r') as f:
#     lines = f.readlines()
#     lines = [line.split()[0] for line in lines]
# with open(base_dir+'film_vocab_most', 'w') as f:
#     f.write('\n'.join(lines))
# %%
re_sub = re.compile('[^\u4e00-\u9fa5]')
def preprocess(line):
    line2 = re_sub.sub('', line)
    out_l = jieba.lcut(line2)
    return out_l
with open(caption_file, 'r') as f:
    lines = f.readlines()
    with Pool(pool_num) as p:
        caption_l = p.map(preprocess, lines)
del lines
caption_pair_l = [[caption_l[i], caption_l[i+1]] for i in range(len(caption_l)-1)]
del caption_l

max_lenth = 0
lens = [len(x[0]) for x in caption_pair_l]
max(lens)
df1 = pd.DataFrame(lens)
# import matplotlib
# %matplotlib inline
df1.describe()

# vocab = {}
# for i in range(len(caption_pair_l)):
#     for x in caption_pair_l[i][0]:
#         if x in vocab:
#             vocab[x] += 1
#         else:
#             vocab[x] = 1
# vocab['_UNK'] = 1000000
# vocab['_PAD'] = 1000000
# vocab['<d>'] = 1000000
# vocab['</d>'] = 1000000
# vocab['<p>'] = 1000000
# vocab['</p>'] = 1000000
# vocab['<s>'] = 1000000
# vocab['</s>'] = 1000000
# len(vocab)
# with open(vocab_file,'w') as f:
#     for x in vocab:
#         f.write(x+' '+str(vocab[x])+'\n')

# %%
def pair2textsumFormat(pair):
    article = ' '.join(pair[0])
    abstract = ' '.join(pair[1])
    article = article + ' | '
    abstract = abstract + ' \n'
    return article + abstract
with Pool(pool_num) as p:
    out_l = p.map(pair2textsumFormat, caption_pair_l)
with open(data_file, 'w') as f:
    f.write(''.join(out_l))
