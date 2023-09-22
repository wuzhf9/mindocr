import math

import mindspore as ms
from mindspore import nn, ops
from mindspore.common.initializer import initializer, HeUniform

__all__ = ["CANHead"]


class ChannelAtt(nn.Cell):
    def __init__(self, channels, reduction):
        super(ChannelAtt, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.SequentialCell([
            nn.Dense(channels, channels//reduction),
            nn.ReLU(),
            nn.Dense(channels//reduction, channels),
            nn.Sigmoid()
        ])

    def construct(self, x):
        B, C, _, _ = x.shape
        y = ops.reshape(self.avg_pool(x), (B, C))
        y = ops.reshape(self.fc(y), (B, C, 1, 1))
        return x * y
    

class CountingDecoder(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(CountingDecoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.trans_layer = nn.SequentialCell([
            nn.Conv2d(in_channels, 512, kernel_size=kernel_size, pad_mode="pad", padding=kernel_size//2),
            nn.BatchNorm2d(512)
        ])
        self.channel_att = ChannelAtt(512, 16)
        self.pred_layer = nn.SequentialCell([
            nn.Conv2d(512, out_channels, kernel_size=1, pad_mode="pad"),
            nn.Sigmoid()
        ])

    def construct(self, x, mask):
        B, _, H, W = x.shape
        x = self.trans_layer(x)
        x = self.channel_att(x)
        x = self.pred_layer(x)
        if mask is not None:
            x = x * mask
        x = ops.reshape(x, (B, self.out_channels, -1))
        x1 = ops.sum(x, dim=-1)
        return x1, ops.reshape(x, (B, self.out_channels, H, W))
    

class PositionEmbeddingSine(nn.Cell):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super(PositionEmbeddingSine, self).__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed.")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def construct(self, mask):
        y_embed = ops.cumsum(mask, axis=1, dtype=ms.float32)
        x_embed = ops.cumsum(mask, axis=2, dtype=ms.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        
        dim_t = ops.arange(self.num_pos_feats, dtype=ms.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = ops.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), axis=4).flatten(start_dim=3)
        pos_y = ops.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), axis=4).flatten(start_dim=3)
        pos = ops.concat((pos_y, pos_x), axis=3).permute(0, 3, 1, 2)
        return pos


class Attention(nn.Cell):
    def __init__(self, hidden_size, attention_dim):
        super(Attention, self).__init__()
        self.hidden_weight = nn.Dense(hidden_size, attention_dim)
        self.attention_conv = nn.Conv2d(1, 512, kernel_size=11, pad_mode="pad", padding=5)
        self.attention_weight = nn.Dense(512, attention_dim, has_bias=False)
        self.alpha_convert = nn.Dense(attention_dim, 1)

    def construct(self, cnn_features, cnn_features_trans, hidden, alpha_sum, image_mask=None):
        query = self.hidden_weight(hidden)
        alpha_sum_trans = self.attention_conv(alpha_sum)
        coverage_alpha = self.attention_weight(alpha_sum_trans.permute(0, 2, 3, 1))
        alpha_score = ops.tanh(query[:, None, None, :] + coverage_alpha + cnn_features_trans.permute(0, 2, 3, 1))
        energy = self.alpha_convert(alpha_score)
        energy = energy - energy.max()
        energy_exp = ops.exp(energy.squeeze(-1))
        if image_mask is not None:
            energy_exp = energy_exp * image_mask.squeeze(1)
        alpha = energy_exp / (ops.sum(energy_exp, dim=(1, 2), keepdim=True) + 1e-10)
        alpha_sum = alpha[:, None, :, :] + alpha_sum
        context_vector = ops.sum(alpha[:, None, :, :] * cnn_features, dim=(2, 3))
        return context_vector, alpha, alpha_sum


class AttDecoder(nn.Cell):
    def __init__(
        self, input_size, hidden_size, out_channels, attention_dim,
        word_num, counting_num, ratio, word_conv_kernel, dropout=0.0
    ):
        super(AttDecoder, self).__init__()
        self.word_num = word_num
        self.ratio = ratio

        self.init_weight = nn.Dense(out_channels, hidden_size)
        self.embedding = nn.Embedding(word_num, input_size)
        self.word_input_gru = nn.GRUCell(input_size, hidden_size)
        self.word_attention = Attention(hidden_size, attention_dim)
        self.encoder_feature_conv = nn.Conv2d(out_channels, attention_dim, kernel_size=word_conv_kernel,
                                              pad_mode="pad", padding=word_conv_kernel//2)
        
        self.word_state_weight = nn.Dense(hidden_size, hidden_size)
        self.word_embedding_weight = nn.Dense(input_size, hidden_size)
        self.word_context_weight = nn.Dense(out_channels, hidden_size)
        self.counting_context_weight = nn.Dense(counting_num, hidden_size)
        self.word_convert = nn.Dense(hidden_size, word_num)

        self.pos_embedding = PositionEmbeddingSine(256, normalize=True)
        self.dropout = nn.Dropout(p=dropout)

    def _init_hidden(self, features, features_mask):
        average = ops.sum(features * features_mask, dim=(2, 3)) / ops.sum(features_mask, dim=(2, 3))
        average = self.init_weight(average)
        return ops.tanh(average)

    def construct(self, cnn_features, labels, counting_preds, images_mask, is_train=True):
        batch_size, num_steps = labels.shape
        height, width = cnn_features.shape[2:]

        images_mask = images_mask[:, :, ::self.ratio, ::self.ratio]
        pos_embedding = self.pos_embedding(images_mask[:, 0, :, :])
        word_probs = ops.zeros((batch_size, num_steps, self.word_num))
        word_alpha_sum = ops.zeros((batch_size, 1, height, width))

        hidden = self._init_hidden(cnn_features, images_mask)
        counting_context_weighted = self.counting_context_weight(counting_preds)
        cnn_features_trans = self.encoder_feature_conv(cnn_features)
        cnn_features_trans = cnn_features_trans + pos_embedding

        word = ops.ones(batch_size, dtype=ms.int32)
        for i in range(num_steps):
            word_embedding = self.embedding(word)
            hidden = self.word_input_gru(word_embedding, hidden)
            word_context_vec, _, word_alpha_sum = self.word_attention(cnn_features, cnn_features_trans, hidden,
                                                                      word_alpha_sum, images_mask)

            current_state = self.word_state_weight(hidden)
            word_weighted_embedding = self.word_embedding_weight(word_embedding)
            word_context_weighted = self.word_context_weight(word_context_vec)

            word_out_state = current_state + word_weighted_embedding + word_context_weighted + counting_context_weighted
            word_out_state = self.dropout(word_out_state)

            word_prob = self.word_convert(word_out_state)
            word_probs[:, i] = word_prob

            if is_train:
                word = labels[:, i]
            else:
                word = ops.argmax(word_prob, dim=1)

        return word_probs


class CANHead(nn.Cell):
    def __init__(
        self, in_channels, out_channels, ratio, attdecoder_args, **kwargs
    ):
        super(CANHead, self).__init__()
        self.ratio = ratio

        self.counting_decoder1 = CountingDecoder(in_channels, out_channels, kernel_size=3)
        self.counting_decoder2 = CountingDecoder(in_channels, out_channels, kernel_size=5)
        self.decoder = AttDecoder(**attdecoder_args, ratio=ratio)

        self._init_weights()

    def _init_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(
                    initializer(HeUniform(math.sqrt(5)), cell.weight.shape, cell.weight.dtype)
                )
            elif isinstance(cell, nn.Dense):
                cell.weight.set_data(
                    initializer(HeUniform(math.sqrt(5)), cell.weight.shape, cell.weight.dtype)
                )

    def construct(self, cnn_features, other_inputs, is_train=True):
        images_mask, labels = other_inputs
        counting_mask = images_mask[:, :, ::self.ratio, ::self.ratio]
        
        counting_preds1, _ = self.counting_decoder1(cnn_features, counting_mask)
        counting_preds2, _ = self.counting_decoder2(cnn_features, counting_mask)
        counting_preds = (counting_preds1 + counting_preds2) / 2

        word_probs = self.decoder(cnn_features, labels, counting_preds, images_mask, is_train=is_train)

        return word_probs, counting_preds, counting_preds1, counting_preds2
