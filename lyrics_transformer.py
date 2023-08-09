import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import os
import re
from transformers import BertTokenizer, TFBertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

class LyricsGeneratorModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout, name="lyrics_generator"):
        super(LyricsGeneratorModel, self).__init__(name=name)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.transformer_blocks = [self.build_transformer_block(embedding_dim, num_heads, ff_dim, dropout) for _ in range(num_transformer_blocks)]
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(units, activation="relu") for units in mlp_units
        ])
        self.lyrics_layer = tf.keras.layers.Dense(vocab_size, activation="softmax")

    def call(self, inputs):
        x = self.embedding(inputs)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        x = tf.reduce_mean(x, axis=1)  # Global average pooling
        x = self.mlp(x)
        return self.lyrics_layer(x)

    def build_transformer_block(self, embedding_dim, num_heads, ff_dim, dropout):
        inputs = tf.keras.Input(shape=(None, embedding_dim))
        x = inputs
        for _ in range(num_heads):
            x = tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=embedding_dim, dropout=dropout
            )(x, x)
        x = tf.keras.layers.Dropout(dropout)(x)
        res = x + inputs
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(res)
        x = tf.keras.layers.Dense(ff_dim, activation="relu")(x)
        x = tf.keras.layers.Dense(embedding_dim)(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        return tf.keras.Model(inputs=inputs, outputs=x)

#USER INPUT
input_lyric_file = 'INPUT FILE NAME'
start_phrase = 'INPUT STARTING PHRASE'
num_epochs = 10 #modify to desired number

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

vocab_size = len(vocab)
embedding_dim = 256
num_heads = 2
ff_dim = 32
num_transformer_blocks = 2
mlp_units = [128]
dropout = 0.1

lyrics_model = LyricsGeneratorModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    ff_dim=ff_dim,
    num_transformer_blocks=num_transformer_blocks,
    mlp_units=mlp_units,
    dropout=dropout
)

lyrics_model.compile(
    loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["sparse_categorical_accuracy"]
)

# Train the model
lyrics_model.fit(dataset, epochs=num_epochs)
