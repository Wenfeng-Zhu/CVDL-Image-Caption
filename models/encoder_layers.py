from torch import nn, Tensor
from torch.nn import MultiheadAttention


class CNNFeedForward(nn.Module):
    def __init__(self, encode_size: int, embed_dim: int, feedforward_dim: int, dropout: float):
        super(CNNFeedForward, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=encode_size,
                               out_channels=feedforward_dim,
                               kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=feedforward_dim,
                               out_channels=encode_size,
                               kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        output = self.conv2(self.relu(self.conv1(inputs.permute(1, 0, 2))))
        output = self.dropout(output)  # type: Tensor
        return self.layer_norm(output.permute(1, 0, 2) + inputs)


class MultiHeadAttention(nn.Module):

    def __init__(self, img_embed_dim: int, num_heads: int, dropout: float):
        super(MultiHeadAttention, self).__init__()

        self.multi_head_attn = MultiheadAttention(embed_dim=img_embed_dim,
                                                  num_heads=num_heads,
                                                  dropout=dropout)
        self.layer_norm = nn.LayerNorm(img_embed_dim)

    def forward(self, enc_inputs: Tensor) -> Tensor:
        enc_outputs, _ = self.multi_head_attn(enc_inputs, enc_inputs, enc_inputs)  # Q, K, V
        enc_outputs = enc_outputs + enc_inputs
        enc_outputs = self.layer_norm(enc_outputs)

        return enc_outputs


class EncoderLayer(nn.Module):

    def __init__(self, img_encode_size: int, img_embed_dim: int,
                 feedforward_dim: int, num_heads: int, dropout: float):
        super(EncoderLayer, self).__init__()

        self.enc_self_attn = MultiHeadAttention(img_embed_dim=img_embed_dim,
                                                num_heads=num_heads,
                                                dropout=dropout)
        self.cnn_ff = CNNFeedForward(encode_size=img_encode_size,
                                     embed_dim=img_embed_dim,
                                     feedforward_dim=feedforward_dim,
                                     dropout=dropout)

    def forward(self, enc_inputs: Tensor) -> Tensor:
        enc_outputs = self.enc_self_attn(enc_inputs)
        enc_outputs = self.cnn_ff(enc_outputs)
        return enc_outputs
