import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    # Implementing the attention module of Bahdanau et al. 2015 where
    # score(h_j, s_(i-1)) = v . tanh(W_1 h_j + W_2 s_(i-1))
    def __init__(self, encoder_hidden_state_dim, decoder_hidden_state_dim, internal_dim=None):
        super(AdditiveAttention, self).__init__()

        if internal_dim is None:
            internal_dim = int((encoder_hidden_state_dim + decoder_hidden_state_dim) / 2)

        self.w1 = nn.Linear(encoder_hidden_state_dim, internal_dim, bias=False)
        self.w2 = nn.Linear(decoder_hidden_state_dim, internal_dim, bias=False)
        self.v = nn.Linear(internal_dim, 1, bias=False)

    def score(self, encoder_state, decoder_state):
        # encoder_state is of shape (batch, enc_dim)
        # decoder_state is of shape (batch, dec_dim)
        # return value should be of shape (batch, 1)
        return self.v(torch.tanh(self.w1(encoder_state) + self.w2(decoder_state)))

    def forward(self, encoder_states, decoder_state):
        # encoder_states is of shape (batch, num_enc_states, enc_dim)
        # decoder_state is of shape (batch, dec_dim)
        score_vec = torch.cat([self.score(encoder_states[:, i], decoder_state) for i in range(encoder_states.shape[1])],
                              dim=1)
        # score_vec is of shape (batch, num_enc_states)

        attention_probs = torch.unsqueeze(F.softmax(score_vec, dim=1), dim=2)
        # attention_probs is of shape (batch, num_enc_states, 1)

        final_context_vec = torch.sum(attention_probs * encoder_states, dim=1)
        # final_context_vec is of shape (batch, enc_dim)

        return final_context_vec, attention_probs


class TemporallyBatchedAdditiveAttention(AdditiveAttention):
    # Implementing the attention module of Bahdanau et al. 2015 where
    # score(h_j, s_(i-1)) = v . tanh(W_1 h_j + W_2 s_(i-1))
    def __init__(self, encoder_hidden_state_dim, decoder_hidden_state_dim, internal_dim=None):
        super(TemporallyBatchedAdditiveAttention, self).__init__(encoder_hidden_state_dim,
                                                                 decoder_hidden_state_dim,
                                                                 internal_dim)

    def score(self, encoder_state, decoder_state):
        # encoder_state is of shape (batch, num_enc_states, max_time, enc_dim)
        # decoder_state is of shape (batch, max_time, dec_dim)
        # return value should be of shape (batch, num_enc_states, max_time, 1)
        return self.v(torch.tanh(self.w1(encoder_state) + torch.unsqueeze(self.w2(decoder_state), dim=1)))

    def forward(self, encoder_states, decoder_state):
        # encoder_states is of shape (batch, num_enc_states, max_time, enc_dim)
        # decoder_state is of shape (batch, max_time, dec_dim)
        score_vec = self.score(encoder_states, decoder_state)
        # score_vec is of shape (batch, num_enc_states, max_time, 1)

        attention_probs = F.softmax(score_vec, dim=1)
        # attention_probs is of shape (batch, num_enc_states, max_time, 1)

        final_context_vec = torch.sum(attention_probs * encoder_states, dim=1)
        # final_context_vec is of shape (batch, max_time, enc_dim)

        return final_context_vec, torch.squeeze(torch.transpose(attention_probs, 1, 2), dim=3)
