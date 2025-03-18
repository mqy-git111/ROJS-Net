import torch
import torch.nn as nn


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, N, M):
        super(LearnedPositionalEncoding, self).__init__()

        self.position_embeddings = nn.Parameter(torch.zeros(1, N, M))  # 8x

    def forward(self, x):
        x = x + self.position_embeddings
        return x


if __name__ == '__main__':
    pos_encode = LearnedPositionalEncoding(None, None, None)
    x = torch.randn((1, 512, 16, 16, 16))
    pos_encode(x, "train")
