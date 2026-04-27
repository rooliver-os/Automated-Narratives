import torch
import torch.nn as nn
import torch.nn.functional as F

class GameNarrativeModel(nn.Module):
    def __init__(self, num_numeric_features, vocab_size, embed_dim=128, hidden_dim=256):
        super(GameNarrativeModel, self).__init__()

        # Encode structured game statistics
        self.stats_encoder = nn.Sequential(
            nn.Linear(num_numeric_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, hidden_dim),
            nn.ReLU()
        )

        # Word embedding for generated text
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Decoder generates the narrative word-by-word
        self.lstm = nn.LSTM(
            input_size=embed_dim + hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )

        # Predict next word
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, stats, captions):
        """
        stats: tensor of game statistics
        captions: tokenized target narrative input
        """

        # Encode game stats
        stats_context = self.stats_encoder(stats)

        # Embed caption words
        embedded = self.embedding(captions)

        # Repeat stats context across every word position
        seq_len = embedded.size(1)
        stats_context = stats_context.unsqueeze(1).repeat(1, seq_len, 1)

        # Combine word embeddings with game stats
        decoder_input = torch.cat((embedded, stats_context), dim=2)

        # Generate hidden states
        lstm_out, _ = self.lstm(decoder_input)

        # Predict vocabulary distribution
        outputs = self.output_layer(lstm_out)

        return outputs
