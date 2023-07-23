import glob
import pickle
import numpy as np
from music21 import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.utils import np_utils

# Installing dependencies
!pip install music21

# Creating an empty list to hold the notes
notes = []

# Loop through each MIDI file and extract the notes from it
for file in glob.glob("lofi-samples/samples/*.mid"):
    midi = converter.parse(file)
    notes_to_parse = midi.flat.notes

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))

# Preparing the dataset
sequence_length = 20
pitchnames = sorted(set(item for item in notes))
note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

network_input = []
network_output = []

for i in range(0, len(notes) - sequence_length, 1):
    sequence_in = notes[i:i + sequence_length]
    sequence_out = notes[i + sequence_length]
    network_input.append([note_to_int[char] for char in sequence_in])
    network_output.append(note_to_int[sequence_out])

n_patterns = len(network_input)
network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
n_vocab = len(set(notes))
network_input = network_input / float(n_vocab)
network_output = np_utils.to_categorical(network_output)

# Building the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dense(n_vocab))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# Training the model
model.fit(network_input, network_output, epochs=20, batch_size=64)

# Generating music
start = np.random.randint(0, len(network_input) - 1)
int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

pattern = network_input[start]
prediction_output = []

for note_index in range(100):
    prediction_input = np.reshape(pattern, (1, len(pattern), 1))
    prediction_input = prediction_input / float(n_vocab)

    prediction = model.predict(prediction_input, verbose=0)
    index = np.argmax(prediction)

    result = int_to_note[index]
    prediction_output.append(result)

    pattern = np.append(pattern, index)
    pattern = pattern[1:]

# Generating the music stream
offset = 0
output_notes = []

for pattern in prediction_output:
    if '.' in pattern or pattern.isdigit():
        notes_in_chord = pattern.split('.')
        notes = []
        for current_note in notes_in_chord:
            new_note = note.Note(int(current_note))
            new_note.storedInstrument = instrument.Piano()
            notes.append(new_note)
        new_chord = chord.Chord(notes)
        new_chord.offset = offset
        output_notes.append(new_chord)
    else:
        new_note = note.Note(pattern)
        new_note.offset = offset
        new_note.storedInstrument = instrument.Piano()
        output_notes.append(new_note)

    offset += 0.5

# Writing the generated music to a MIDI file
s = stream.Stream(output_notes)
mf = s.write('midi', fp="lofi-samples/testOutput.mid")
