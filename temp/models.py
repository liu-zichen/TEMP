from transformers.models.bert import BertTokenizer, BertConfig, BertPreTrainedModel, BertModel
from transformers.models.electra import ElectraPreTrainedModel, ElectraModel, ElectraConfig
from transformers.models.albert import AlbertModel, AlbertPreTrainedModel, AlbertConfig
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel, RobertaConfig
from transformers.models.xlnet import XLNetModel, XLNetPreTrainedModel, XLNetConfig, XLNetForSequenceClassification
from transformers.models.xlm import XLMModel, XLMPreTrainedModel, XLMConfig, XLMForSequenceClassification
from transformers.modeling_utils import SequenceSummary
# from transformers.models.xlm_roberta
# from transformers.models.bart import BartTokenizer, BartPretrainedModel, BartModel, BartConfig
from torch import nn
import torch
from transformers.activations import get_activation


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


class ElectraClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = get_activation("gelu")(x)  # although BERT uses tanh here, it seems Electra authors used gelu here
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class TEMPElectra(ElectraPreTrainedModel):
    EPS = 1e-9

    def __init__(self, config: ElectraConfig):
        super().__init__(config)
        self.electra = ElectraModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        config.num_labels = 1
        self.classifier = ElectraClassificationHead(config)
        # self.classifier = nn.Linear(config.hidden_size, 1)
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
        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
        )
        sequence_output = discriminator_hidden_states[0]
        logits = self.classifier(sequence_output)
        if labels is not None:
            return self.loss_fct(logits, labels)
        return logits

    @classmethod
    def margin_loss_fct(cls, pos_score: torch.Tensor, neg_score: torch.Tensor, margin: torch.Tensor):
        loss = (-(pos_score.squeeze().relu().clamp(min=cls.EPS)) +
                neg_score.squeeze().relu().clamp(min=cls.EPS) +
                margin.squeeze().relu().clamp(min=cls.EPS)).clamp(min=0)
        return loss.sum()


class TEMPAlbert(AlbertPreTrainedModel):
    EPS = 1e-9

    def __init__(self, config: AlbertConfig):
        super().__init__(config)
        self.albert = AlbertModel(config)
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
        outputs = self.albert(
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


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class TEMPRoberta(RobertaPreTrainedModel):
    EPS = 1e-9

    def __init__(self, config: RobertaConfig):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        config.num_labels = 1
        self.classifier = RobertaClassificationHead(config)
        # self.classifier = nn.Linear(config.hidden_size, 1)
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
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
        if labels is not None:
            return self.loss_fct(logits, labels)
        return logits

    @classmethod
    def margin_loss_fct(cls, pos_score: torch.Tensor, neg_score: torch.Tensor, margin: torch.Tensor):
        loss = (-(pos_score.squeeze().relu().clamp(min=cls.EPS)) +
                neg_score.squeeze().relu().clamp(min=cls.EPS) +
                margin.squeeze().relu().clamp(min=cls.EPS)).clamp(min=0)
        return loss.sum()


class TEMPXLNet(XLNetPreTrainedModel):
    EPS = 1e-9

    def __init__(self, config: XLNetConfig):
        config.num_labels = 1
        super().__init__(config)
        self.xlnet = XLNetModel(config)
        self.sequence_summary = SequenceSummary(config)
        self.classifier = nn.Linear(config.d_model, 1)
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
        outputs = self.xlnet(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
        )
        pooled_output = outputs[0]
        output = self.sequence_summary(pooled_output)
        logits = self.classifier(output)
        if labels is not None:
            return self.loss_fct(logits, labels)
        return logits

    @classmethod
    def margin_loss_fct(cls, pos_score: torch.Tensor, neg_score: torch.Tensor, margin: torch.Tensor):
        loss = (-(pos_score.squeeze().relu().clamp(min=cls.EPS)) +
                neg_score.squeeze().relu().clamp(min=cls.EPS) +
                margin.squeeze().relu().clamp(min=cls.EPS)).clamp(min=0)
        return loss.sum()


class TEMPXLM(XLMPreTrainedModel):
    EPS = 1e-9

    def __init__(self, config: XLMConfig):
        config.num_labels = 1
        super().__init__(config)
        self.xlnet = XLMModel(config)
        self.sequence_summary = SequenceSummary(config)
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
        outputs = self.xlnet(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
        )
        pooled_output = outputs[0]
        output = self.sequence_summary(pooled_output)
        logits = output.view(-1, 1)
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
