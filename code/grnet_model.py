import torch
import torch.nn as nn
from better_lstm_model import LSTM

class AttentionWeights(nn.Module):
    def __init__(self, max_dim):
        super(AttentionWeights, self).__init__()
        self.max_dim = max_dim
        self.attention = nn.Linear(446, 1)  # 446 is LSTM hidden size

    def forward(self, lstm_output):
        # lstm_output shape: (batch_size, seq_len, hidden_size)
        attention_scores = self.attention(lstm_output)  # (batch_size, seq_len, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        return attention_weights


class ContextVector(nn.Module):
    def __init__(self):
        super(ContextVector, self).__init__()

    def forward(self, lstm_output, attention_weights):
        # lstm_output: (batch_size, seq_len, hidden_size)
        # attention_weights: (batch_size, seq_len, 1)
        context = torch.sum(
            lstm_output * attention_weights, dim=1
        )  # (batch_size, hidden_size)
        return context

class PlanModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        goal_size,
        max_dim,
        embedding_dim=85,
        lstm_hidden=446,
        dropouti=0.2,
        dropoutw=0.2,
        dropouto=0.2,
    ):
        super(PlanModel, self).__init__()
        self.max_dim = max_dim

        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=0)
        
        self.lstm = LSTM(
            embedding_dim, lstm_hidden, batch_first=True, dropouti=dropouti, dropoutw=dropoutw, dropouto=dropouto)
        
        self.attention_weights = AttentionWeights(max_dim)
        self.context_vector = ContextVector()
        self.output_layer = nn.Linear(lstm_hidden, goal_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)

        # Create mask for padded elements
        mask = (x != 0).float()  # (batch_size, seq_len)
        if len(embedded.shape) > 3:
            print(embedded.shape)
            print(embedded)
        lstm_output, _ = self.lstm(embedded)  # (batch_size, seq_len, lstm_hidden)

        # Apply mask to LSTM output
        lstm_output = lstm_output * mask.unsqueeze(-1)
        
        attention_weights = self.attention_weights(
            lstm_output
        )  # (batch_size, seq_len, 1)
        context = self.context_vector(
            lstm_output, attention_weights
        )  # (batch_size, lstm_hidden)

        output = self.output_layer(context)  # (batch_size, goal_size)
        output = self.sigmoid(output)

        return output