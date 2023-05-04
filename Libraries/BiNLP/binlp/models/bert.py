# coding=utf-8
# 2020.04.20 - Add&replace quantization modules
#              Huawei Technologies Co., Ltd <zhangwei379@huawei.com>
# Copyright (c) 2020, Huawei Technologies Co., Ltd.  All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

from __future__ import absolute_import, division, print_function, unicode_literals

import pdb
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
import sys

import logging
import math
import os

import torch
from torch import nn
from torch.autograd import Variable
from .configuration import BertConfig
from torch.nn import Parameter
from .builder import TRANSFORMERS
from mmcv.cnn import (build_activation_layer, build_conv_layer)

logger = logging.getLogger(__name__)

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"
cnt_epoch = -1
last_epoch = -1

g_num = 0

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class BinaryQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input[0].ge(1)] = 0
        grad_input[input[0].le(-1)] = 0
        return grad_input


class ZMeanBinaryQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        out[out==-1] = 0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input[0].ge(1)] = 0
        grad_input[input[0].le(-1)] = 0
        return 


class QuantizeEmbedding(nn.Embedding):
    def __init__(self,  *kargs, padding_idx=None, config=None):
        super(QuantizeEmbedding, self).__init__(*kargs, padding_idx=padding_idx)
        self.weight_quantizer = BinaryQuantizer

    def forward(self, input, type=None):
        scaling_factor = torch.mean(abs(self.weight), dim=1, keepdim=True)
        scaling_factor = scaling_factor.detach()
        real_weights = self.weight - torch.mean(self.weight, dim=-1, keepdim=True)
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        weight = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        out = nn.functional.embedding(
            input, weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)
        return out
    

class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        if config.linear_cfg.type != 'Linear':
            self.word_embeddings = QuantizeEmbedding(config.vocab_size,
                                                    config.hidden_size,
                                                    padding_idx=0)
        else:
            self.word_embeddings = nn.Embedding(config.vocab_size,
                                            config.hidden_size,
                                            padding_idx=0)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size)
        
        self.LayerNorm = nn.LayerNorm(config.hidden_size,
                                      eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length,
                                    dtype=torch.long,
                                    device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BinaryBertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BinaryBertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" %
                (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = build_conv_layer(config.linear_cfg,
                                      config.hidden_size,
                                      self.all_head_size)
        self.key = build_conv_layer(config.linear_cfg,
                                      config.hidden_size,
                                      self.all_head_size)
        self.value = build_conv_layer(config.linear_cfg,
                                      config.hidden_size,
                                      self.all_head_size)
        prob_mean = None
        self.act_quantizer = BinaryQuantizer

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                       self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self,
                hidden_states,
                attention_mask,
                output_att=False):

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        query_scores = torch.matmul(query_layer, query_layer.transpose(-1, -2))
        query_scores = query_scores / math.sqrt(self.attention_head_size)

        key_scores = torch.matmul(key_layer, key_layer.transpose(-1, -2))
        key_scores = key_scores / math.sqrt(self.attention_head_size)

        query_layer = self.act_quantizer.apply(query_layer)
        key_layer = self.act_quantizer.apply(key_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = self.dropout(attention_scores)

        value_scores = torch.matmul(value_layer, value_layer.transpose(-1, -2))
        value_scores = value_scores / math.sqrt(self.attention_head_size)

        attention_probs = ZMeanBinaryQuantizer.apply(attention_probs)
        value_layer = self.act_quantizer.apply(value_layer)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size, )
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_scores, value_scores, 0, query_scores, key_scores


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" %
                (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
 
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                       self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # distil qxk
        attention_scores = torch.matmul(query_layer, key_layer.transpose(
            -1, -2))
        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size)  

        if attention_mask is not None:                                # changed here
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        
        # distill qxq
        query_scores = torch.matmul(query_layer, query_layer.transpose(-1, -2))
        query_scores = query_scores / math.sqrt(self.attention_head_size)

        # distill kxk
        key_scores = torch.matmul(key_layer, key_layer.transpose(-1, -2))
        key_scores = key_scores / math.sqrt(self.attention_head_size)

        # distil vxv
        value_scores = torch.matmul(value_layer, value_layer.transpose(-1, -2))
        value_scores = value_scores / math.sqrt(self.attention_head_size)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size, )
        context_layer = context_layer.view(*new_context_layer_shape)
        context_score = 0
        return context_layer, attention_scores, value_scores, context_score, query_scores, key_scores


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = build_conv_layer(config.linear_cfg, config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    
    
class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        if config.linear_cfg.type != 'Linear':
            self.self = BinaryBertSelfAttention(config)
        else:
            self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output, layer_att, value_att, context_score, query_scores, key_scores = self.self(
            input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, layer_att, value_att, context_score, query_scores, key_scores


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = build_conv_layer(
            config.linear_cfg, config.hidden_size, config.intermediate_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = gelu(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = build_conv_layer(
            config.linear_cfg, config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size,
                                      eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output, layer_att, value_att, context_score, query_score, key_score = self.attention(
            hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        return layer_output, layer_att, value_att, context_score, query_score, key_score


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.layer = nn.ModuleList(
            [BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask):
        all_encoder_layers = [hidden_states]
        all_encoder_atts = []
        all_value_atts = []
        all_context_scores = []
        all_query_scores = []
        all_key_scores = []

        for _, layer_module in enumerate(self.layer):
            hidden_states, layer_att, value_att, context_score, query_score, key_score = layer_module(
                hidden_states, attention_mask)
            all_encoder_layers.append(hidden_states)
            all_encoder_atts.append(layer_att)
            all_value_atts.append(value_att)
            all_context_scores.append(context_score)
            all_query_scores.append(query_score)
            all_key_scores.append(key_score)

        return all_encoder_layers, all_encoder_atts, all_value_atts, all_context_scores, all_query_scores, all_key_scores


class BertPooler(nn.Module):
    def __init__(self, config, recurs=None):
        super(BertPooler, self).__init__()
        self.dense = build_conv_layer(
            config.linear_cfg, config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        pooled_output = hidden_states[-1][:, 0]
        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)

        return pooled_output


class BertPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, *args, **kwargs):
        super(BertPreTrainedModel, self).__init__()
        self.config = config
        
    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0,
                                       std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_path, linear_cfg, num_labels):
        config_file = os.path.join(pretrained_path, CONFIG_NAME)
        config = BertConfig.from_json_file(config_file)
        setattr(config, 'linear_cfg', linear_cfg)
        logger.info("Model config {}".format(config))
        # Instantiate model.

        model = cls(config, num_labels=num_labels)
        weights_path = os.path.join(pretrained_path, WEIGHTS_NAME)
        logger.info("Loading model {}".format(weights_path))
        state_dict = torch.load(weights_path, map_location='cpu')

        # Load from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(
                prefix[:-1], {})
            module._load_from_state_dict(state_dict, prefix, local_metadata,
                                         True, missing_keys, unexpected_keys,
                                         error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        start_prefix = ''
        if not hasattr(model, 'bert') and any(
                s.startswith('bert.') for s in state_dict.keys()):
            start_prefix = 'bert.'

        logger.info('loading model...')
        load(model, prefix=start_prefix)
        logger.info('done!')
        if len(missing_keys) > 0:
            logger.info(
                "Weights of {} not initialized from pretrained model: {}".
                format(model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info(
                "Weights from pretrained model not used in {}: {}".format(
                    model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError(
                'Error(s) in loading state_dict for {}:\n\t{}'.format(
                    model.__class__.__name__, "\n\t".join(error_msgs)))

        return model


class BertModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers, attention_scores, value_scores, context_scores, query_scores, key_scores = self.encoder(
            embedding_output, extended_attention_mask)

        pooled_output = self.pooler(encoded_layers)
        return encoded_layers, attention_scores, pooled_output, value_scores, context_scores, query_scores, key_scores


@TRANSFORMERS.register_module()
class BertSeqClassifier(BertPreTrainedModel):
    def __init__(self, config, num_labels=2):
        super(BertSeqClassifier, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                labels=None):
        encoded_layers, attention_scores, pooled_output, value_scores, context_scores, query_scores, key_scores = self.bert(
            input_ids, token_type_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, encoded_layers, value_scores, query_scores, key_scores
        else:
            return logits, encoded_layers, value_scores, query_scores, key_scores
