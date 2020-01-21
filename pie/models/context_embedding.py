import torch.nn as nn


class ReducedTransformer(nn.Module):
    def __init__(self, transformer, transformer_dim, out_dim):
        self.transformer = transformer
        self.transformer_dim: int = transformer_dim
        self.out_dim: int = out_dim

        self.linear = nn.Linear(self.transformer_dim, self.out_dim)

    def forward(self, sentences):
        """

        :param sentences: Batch of sentences with words
        :returns: Embedding, Sentence
        """

        words, sentence = self.transformer(sentences)
        words = self.linear(words)
        return words, sentence
