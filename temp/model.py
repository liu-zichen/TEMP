from transformers.models.bert import BertTokenizer, BertConfig, BertPreTrainedModel, BertModel
from transformers.models.electra import ElectraTokenizer, ElectraPreTrainedModel, ElectraModel, ElectraConfig
from transformers.models.bart import BartTokenizer, BartPretrainedModel, BartModel, BartConfig
from torch import nn
import torch


class TEMPBert(BertPreTrainedModel):
    EPS = 1e-9

    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_fct = nn.BCEWithLogitsLoss()
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            head_mask=None,
            labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if labels is not None:
            return self.loss_fct(logits, labels)
        return logits

    @classmethod
    def margin_loss_fct(cls, pos_score: torch.Tensor, neg_score: torch.Tensor, margin: torch.Tensor):
        loss = (-(pos_score.squeeze().relu().clamp(min=cls.EPS)) +
                neg_score.squeeze().relu().clamp(min=cls.EPS) +
                margin.squeeze().relu().clamp(min=cls.EPS)).clamp(min=0)
        return loss.sum()


class TEMPElectra(ElectraPreTrainedModel):
    EPS = 1e-9

    def __init__(self, config: ElectraConfig):
        super().__init__(config)
        self.electra = ElectraModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_fct = nn.BCEWithLogitsLoss()
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            head_mask=None,
            labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if labels is not None:
            return self.loss_fct(logits, labels)
        return logits

    @classmethod
    def margin_loss_fct(cls, pos_score: torch.Tensor, neg_score: torch.Tensor, margin: torch.Tensor):
        loss = (-(pos_score.squeeze().relu().clamp(min=cls.EPS)) +
                neg_score.squeeze().relu().clamp(min=cls.EPS) +
                margin.squeeze().relu().clamp(min=cls.EPS)).clamp(min=0)
        return loss.sum()


# class TEMPBart(BartPretrainedModel):
#     EPS = 1e-9
#
#     def __init__(self, config: BartConfig):
#         super().__init__(config)
#         self.electra = BartModel(config)
#         # self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, 1)
#         self.loss_fct = nn.BCEWithLogitsLoss()
#         self.init_weights()
#
#     def forward(
#             self,
#             input_ids=None,
#             attention_mask=None,
#             token_type_ids=None,
#             head_mask=None,
#             labels=None,
#     ):
#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             head_mask=head_mask,
#         )
#         pooled_output = outputs[1]
#         # pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)
#         if labels is not None:
#             return self.loss_fct(logits, labels)
#         return logits
#
#     @classmethod
#     def margin_loss_fct(cls, pos_score: torch.Tensor, neg_score: torch.Tensor, margin: torch.Tensor):
#         loss = (-(pos_score.squeeze().relu().clamp(min=cls.EPS)) +
#                 neg_score.squeeze().relu().clamp(min=cls.EPS) +
#                 margin.squeeze().relu().clamp(min=cls.EPS)).clamp(min=0)
#         return loss.sum()
