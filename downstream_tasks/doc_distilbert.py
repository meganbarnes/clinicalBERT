import copy
import logging
import math

import numpy as np
import torch
import torch.nn as nn

from transformers import PreTrainedModel, DistilBertModel, DistilBertConfig

DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "distilbert-base-uncased": "https://cdn.huggingface.co/distilbert-base-uncased-pytorch_model.bin",
    "distilbert-base-uncased-distilled-squad": "https://cdn.huggingface.co/distilbert-base-uncased-distilled-squad-pytorch_model.bin",
    "distilbert-base-cased": "https://cdn.huggingface.co/distilbert-base-cased-pytorch_model.bin",
    "distilbert-base-cased-distilled-squad": "https://cdn.huggingface.co/distilbert-base-cased-distilled-squad-pytorch_model.bin",
    "distilbert-base-german-cased": "https://cdn.huggingface.co/distilbert-base-german-cased-pytorch_model.bin",
    "distilbert-base-multilingual-cased": "https://cdn.huggingface.co/distilbert-base-multilingual-cased-pytorch_model.bin",
    "distilbert-base-uncased-finetuned-sst-2-english": "https://cdn.huggingface.co/distilbert-base-uncased-finetuned-sst-2-english-pytorch_model.bin",
}

# INTERFACE FOR ENCODER AND TASK SPECIFIC MODEL #
class DistilBertPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = DistilBertConfig
    pretrained_model_archive_map = DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = None
    base_model_prefix = "distilbert"

    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, nn.Embedding):
            if module.weight.requires_grad:
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

DISTILBERT_START_DOCSTRING = r"""

    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config (:class:`~transformers.DistilBertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

DISTILBERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`transformers.DistilBertTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.encode_plus` for details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.

            `What are attention masks? <../glossary.html#attention-mask>`__
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            :obj:`1` indicates the head is **not masked**, :obj:`0` indicates the head is **masked**.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
"""

@add_start_docstrings(
    """DistilBert Model transformer with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks. """,
    DISTILBERT_START_DOCSTRING,
)
class DocDistilBertForSequenceClassification(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        self.lstm = nn.LSTM(config.dim,config.dim,batch_first=True)

        # dropout = nn.Dropout(p=bert_model_config.hidden_dropout_prob)
        
        # classifier = nn.Sequential(
        #     nn.Dropout(p=bert_model_config.hidden_dropout_prob),
        #     nn.Linear(bert_model_config.hidden_size, bert_model_config.num_labels),
        #     nn.Tanh()
        # )

        self.init_weights()

@add_start_docstrings_to_callable(DISTILBERT_INPUTS_DOCSTRING)
    def forward(self, input_ids=None, attention_mask=None, head_mask=None, inputs_embeds=None, labels=None):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.DistilBertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
        import torch

        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-cased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

        """

        ##########

        input_ids.view((input_ids.shape[0]*input_ids.shape[1],-1))

        distilbert_output = self.distilbert(
            input_ids=input_ids.view((input_ids.shape[0]*input_ids.shape[1],-1)), attention_mask=attention_mask.view((attention_mask.shape[0]*attention_mask.shape[1],-1)), head_mask=head_mask.view((head_mask.shape[0]*head_mask.shape[1],-1)), inputs_embeds=inputs_embeds
        ) # (bs, seq_len, dim)
        hidden_state = distilbert_output[0].view((input_ids.shape[0], input_ids.shape[1], input_ids.shape[2], -1))  # (bs, doc_len, seq_len, dim)
        pooled_output = hidden_state[:,:,0]  # (bs, doc_len, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, doc_len, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, doc_len, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, doc_len, dim)

        output, (_, _) = self.lstm(pooled_output) # (bs, doc_len, dim)

        last_layer = output[:,-1] # (bs, dim)

        logits = self.classifier(last_layer)  # (bs, num_labels)

        outputs = (logits,) + distilbert_output[1:]
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

