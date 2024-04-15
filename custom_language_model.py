# custom_language_model.py


import torch.nn as nn

class CustomLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(CustomLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, hidden=None):
        # Embed the input
        embed = self.embedding(input_ids)

        # Pass through the LSTM
        output, hidden = self.lstm(embed, hidden)

        # Pass through the final layer
        output = self.fc(output)
        return output, hidden
