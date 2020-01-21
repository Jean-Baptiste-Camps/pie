import torch.nn as nn


class ReducedTransformer(nn.Module):
    def __init__(self, transformer, transformer_dim, out_dim):

        self.transformer_dim: int = transformer_dim
        self.out_dim: int = out_dim
        super(ReducedTransformer, self).__init__()

        self.transformer = transformer
        self.linear = nn.Linear(self.transformer_dim, self.out_dim)

    def forward(self, sentences):
        """

        :param sentences: Batch of sentences with words
        :returns: Embedding, Sentence
        """
        words, sentence = self.transformer(sentences)
        words = self.linear(words)
        return words, sentence


def ReducerFunction(transformer, linear):
    def model(sentences):
        words, sentence = transformer(sentences)
        words = linear(words)
        return words, sentence
    return model