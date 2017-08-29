# Neural_Conversation_Models
=================================
This implementation contains an extension of seq2seq tutorial for conversation models in Tensorflow:

1. Option to use Beam Search and Beam Size for decoding

2. Currently, it supports
    - Simple seq2seq  models
    - Attention based seq2seq models

3. To get better results use beam search during decoding / inference

Examples of basic model can be found in this paper.

https://arxiv.org/abs/1702.05512


Usage
--------
    $ CUDA_VISIBLE_DEVICES=3 python neural_conversation_model.py --en_vocab_size=40000 --size=256 --data_path=/home/data/ljc/film_caption/corpus_ncm/film_data --vocab_path=/home/data/ljc/film_caption/corpus_ncm/film_vocab_most --dev_data=/home/data/ljc/film_caption/corpus_ncm/film_data --attention

    $ CUDA_VISIBLE_DEVICES=3 python neural_conversation_model.py --train_dir ubuntu/ --en_vocab_size 1699 --size 512 --data_path ubuntu/my_train_2 --dev_data ubuntu/my_train_2  --vocab_path ubuntu/my_train_2_vocab --attention
    $ CUDA_VISIBLE_DEVICES=3 python neural_conversation_model.py --train_dir ubuntu/ --en_vocab_size 1699 --size 512 --data_path ubuntu/my_train_2 --dev_data ubuntu/my_train_2  --vocab_path ubuntu/my_train_2_vocab --attention --decode --beam_search --beam_size 25

File list
---------------
- read_chat_data_test.py: test the format of text data red in.

Convert to tf 1.0
---------------
- my_seq2seq.py: `from tensorflow.python.ops import rnn` --> `from tensorflow.contrib.rnn.python.ops import core_rnn`
- my_seq2seq.py: `rnn.rnn` --> `core_rnn.static_rnn`
- my_seq2seq.py: `from tensorflow.python.ops.rnn_cell import _linear as linear` --> `from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _linear as linear`
- my_seq2seq.py: `from tensorflow.python.ops import rnn_cell` --> `from tensorflow.contrib.rnn.python.ops import core_rnn_cell as rnn_cell`
- my_seq2seq.py: `op_scope` --> `name_scope`
- my_seq2seq.py: `attention_states = array_ops.concat(1, top_states)` --> `attention_states = array_ops.concat(axis=1, values=top_states)`
- my_seq2seq.py: `array_ops.pack` --> `array_ops.stack`
- my_seq2seq.py: `def embedding_attention_seq2seq(encoder_inputs, decoder_inputs, cell, ...` --> `def embedding_attention_seq2seq(encoder_inputs, decoder_inputs, cell_1,cell_2, ...`



- seq2seq_model.py: `tf.nn.rnn_cell` --> `tf.contrib.rnn`
- seq2seq_model.py: 在tf.nn.sampled_softmax_loss中inputs,labels-->labels, inputs,
- seq2seq_model.py:
```python
single_cell = tf.nn.rnn_cell.GRUCell(size)
if use_lstm:
  single_cell = tf.nn.rnn_cell.BasicLSTMCell(size)
cell = single_cell
if num_layers > 1:
  cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers, state_is_tuple=False)
```
-------->
```python
def gru_cell():
  return tf.contrib.rnn.core_rnn_cell.GRUCell(size, reuse=tf.get_variable_scope().reuse)#tf.get_variable_scope().reuse
def lstm_cell():
  return tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(size, reuse=tf.get_variable_scope().reuse)#tf.get_variable_scope().reuse
single_cell = gru_cell
if use_lstm:
  single_cell = lstm_cell
cell = single_cell()
if num_layers > 1:
  cell_1 = tf.contrib.rnn.core_rnn_cell.MultiRNNCell([single_cell() for _ in range(num_layers)], state_is_tuple=False)
  cell_2 = tf.contrib.rnn.core_rnn_cell.MultiRNNCell([single_cell() for _ in range(num_layers)], state_is_tuple=False)

```
- neural_conversation_model.py: `curr = range(beam_size) ` --> `curr = list(range(beam_size))`
- neural_conversation_model.py: `sentence_to_token_ids(tf.compat.as_bytes(sentence), vocab)` --> `sentence_to_token_ids(sentence, vocab)`
- neural_conversation_model.py: `entities = line.lower().split("\t")` --> `entities = line.lower().split("|")`

Prerequisites
-------------

- Python Python 3.3+ --> Python 3.6
- [NLTK](http://www.nltk.org/)
- [TensorFlow](https://www.tensorflow.org/) 0.12.1 --> TensorFlow 1.1.0

Installations
-----

* Mac
```
virtualenv --no-site-packages -p /usr/local/bin/python3.6 ~/venv-py3
source ~/venv-py3/bin/activate
pip3 install --upgrade \
 https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.12.1-py3-none-any.whl # for CPU Usage
```

* Linux with GPU Driver
```
virtualenv --no-site-packages -p /usr/local/bin/python3.6 ~/venv-py3
source ~/venv-py3/bin/activate
pip3 install --upgrade \ https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.1-cp34-cp34m-linux_x86_64.whl
```

Data
-----
Data accepted is in the tsv format where first component is the context and second is the reply

TSV format Ubuntu Dialog Data can be found [here](https://drive.google.com/file/d/0BwPa9lrosQKdSTZxZ0tydUFGWE0/view) or [Git repo](http://git.oschina.net/ubiware/neural_conversation_models_ubuntu_corpus).

example :-
1. What are you doing ? \t Writing seq2seq model .

Usage
-----

To train a model with Ubuntu dataset:

    $ python neural_conversation_model.py --train_dir ubuntu/ --en_vocab_size 60000 --size 512 --data_path ubuntu/train.tsv --dev_data ubuntu/valid.tsv  --vocab_path ubuntu/60k_vocan.en --attention

To test an existing model:

    $ python neural_conversation_model.py --train_dir ubuntu/ --en_vocab_size 60000 --size 512 --data_path ubuntu/train.tsv --dev_data ubuntu/valid.tsv  --vocab_path ubuntu/60k_vocan.en --attention --decode --beam_search --beam_size 25

Todo
-----
1. Add other state of art neural models.
2. Adding layer normalization( in progress )

https://github.com/pbhatia243/tf-layer-norm

## Contact
Parminder Bhatia, parminder.bhatia243@gmail.com
