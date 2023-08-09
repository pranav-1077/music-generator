import torch
import torch.nn as nn
import torch.optim as optim
from music21 import converter, note, chord
import glob
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class LofiTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super(LofiTransformer, self).__init__()
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers)
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        x = self.embedding(x)
        # Assuming output shape (sequence length, batch size, embedding size)
        output = self.transformer(x, x)
        return output.permute(1, 0, 2)  # Reshape output

# Load and preprocess MIDI data
def process_midi_files():
    notes = []

    for file in glob.glob("/content/drive/MyDrive/lofi_samples/*.mid"):
        midi = converter.parse(file)
        notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

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

    return network_input, network_output, n_vocab

# Preprocess and load the data
network_input, network_output, vocab_size = process_midi_files()

# Create a DataLoader for training
batch_size = 64
train_dataset = TensorDataset(torch.tensor(network_input, dtype=torch.float32), torch.tensor(network_output, dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize the model
d_model = 128
nhead = 4
num_layers = 4
model = LofiTransformer(vocab_size, d_model, nhead, num_layers)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#Training Loop
num_epochs = 100
for epoch in range(num_epochs):
    total_loss = 0
    for batch_input, batch_output in train_loader:
        optimizer.zero_grad()
        batch_input = batch_input.permute(1, 0, 2).long()  # Convert to torch.long
        batch_input = batch_input.squeeze(2).permute(1, 0)  # Reshape
        output = model(batch_input)
        loss = criterion(output.view(-1, vocab_size), batch_output.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}')
