import dataclasses
from abc import ABC

import torch
import transformers
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model,
)


@dataclasses.dataclass
class ModelOutput(transformers.file_utils.ModelOutput):
    logits: torch.FloatTensor = None
    hidden_states: torch.FloatTensor = None


class ModelHead(torch.nn.Module):

    def __init__(self, config):
        super().__init__()

        self.dense = torch.nn.Linear(
            config.hidden_size,  # fit size of transformer output
            config.hidden_size,
        )
        self.dropout = torch.nn.Dropout(
            config.final_dropout,
        )
        self.out_proj = torch.nn.Linear(
            config.hidden_size,
            config.num_labels,  # number of target classes
        )

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class ModelCategorical(Wav2Vec2PreTrainedModel, ABC):

    def __init__(self, config):

        super().__init__(config)

        self.wav2vec2 = Wav2Vec2Model(config)
        self.head_categorical = ModelHead(config)
        self.init_weights()

    def pooling(
            self,
            hidden_states,
            attention_mask,
    ):

        if attention_mask is None:  # for evaluation with batch_size==1
            outputs = torch.mean(hidden_states, dim=1)
        else:
            attention_mask = self._get_feature_vector_attention_mask(
                hidden_states.shape[1],
                attention_mask,
            )
            hidden_states = hidden_states * torch.reshape(
                attention_mask,
                (-1, attention_mask.shape[-1], 1),
            )
            outputs = torch.sum(hidden_states, dim=1)
            attention_sum = torch.sum(attention_mask, dim=1)
            outputs = outputs / torch.reshape(attention_sum, (-1, 1))

        return outputs

    def forward(
            self,
            input_values,
            attention_mask=None,
    ):

        # normalize input signal
        mean = input_values.mean()
        var = torch.square(input_values - mean).mean()
        input_values = (input_values - mean) / torch.sqrt(var + 1e-7)

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
        )

        hidden_states = self.pooling(
            outputs.last_hidden_state,
            attention_mask,
        )
        logits_cat = self.head_categorical(hidden_states)

        return ModelOutput(
            logits=logits_cat.squeeze(),
            hidden_states=hidden_states,
        )
