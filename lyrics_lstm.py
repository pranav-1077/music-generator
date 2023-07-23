import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import os
import re

#USER INPUT
input_lyric_file = 'INSERT LYRICS FILE'
start_phrase = 'INSERT STARTING LYRIC'

# open up rapper lyric file
with open(input_lyric_file,'r') as file:
	text = file.read().lower()
print('text length: ', len(text))

# get chars from lyrics file
vocab = sorted(list(set(text)))
print('total chars: ', len(vocab))

# create a way to index from character to index
char2idx = {u:i for i,u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in text])

# set up sequence lengths 
seq_length = 50
examples_per_epoch = len(text)//(seq_length + 1)

# create training examples
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

# batch method 
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

# create input/target sections
def split_input_target(chunk):
	input_text = chunk[:-1]
	target_text = chunk[1:]
	return input_text, target_text

dataset = sequences.map(split_input_target)

BATCH_SIZE = 1
BUFFER_SIZE = 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# length of vocab in the chars
vocab_size = len(vocab)
embedding_dim = 128
rnn_units = 512

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.LSTM(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.compile(optimizer='adam', loss=loss)

# Directory with checkpoints
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

model.build(tf.TensorShape([1, None]))

model.summary()

history = model.fit(dataset, epochs=7, callbacks=[checkpoint_callback])

def generate_text(model, start_string):

  # Number of characters to generate
  num_generate = 5000

  #vectorize
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  #results
  text_generated = []
  temperature = 1.0

  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      predictions = tf.squeeze(predictions, 0)
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
      input_eval = tf.expand_dims([predicted_id], 0)
      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))
  
#final output
print(generate_text(model, start_string=start_phrase))
