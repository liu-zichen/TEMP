import torch
import temp.sampler as sampler
from torch.utils.data import dataloader
import transformers
from transformers import BertTokenizer, ElectraTokenizer, AlbertTokenizer, RobertaTokenizer, XLMTokenizer, \
    XLNetTokenizer
from tqdm import tqdm
from temp.base import BaseTrainer
import codecs
from temp import TEMPBert, TEMPElectra, TEMPAlbert, TEMPRoberta, TEMPXLNet, TEMPXLM
import json


class Trainer(BaseTrainer):
    models = {'bert': TEMPBert, 'electra': TEMPElectra, 'albert': TEMPAlbert, 'roberta': TEMPRoberta,
              'xlnet': TEMPXLNet, 'xlm': TEMPXLM}
    tokenizers = {'bert': BertTokenizer, 'electra': ElectraTokenizer, 'albert': AlbertTokenizer,
                  'roberta': RobertaTokenizer, 'xlnet': XLNetTokenizer, 'xlm': XLMTokenizer}

    def __init__(self, args):
        super().__init__(args)
        with codecs.open(args.taxo_path, encoding='utf-8') as f:
            # TAXONOMY FILE FORMAT: hypernym <TAB> term
            tax_lines = f.readlines()
        tax_pairs = [line.strip().split("\t") for line in tax_lines]
        self.tax_graph = sampler.TaxStruct(tax_pairs)
        self.sampler = sampler.Sampler(self.tax_graph)

        self.model = self.models[args.model_type].from_pretrained(args.pretrained_path,
                                                                  # gradient_checkpointing=True,
                                                                  output_attentions=False,
                                                                  output_hidden_states=False,
                                                                  cache_dir="./cache/",
                                                                  # force_download=True
                                                                  )
        self._tokenizer = self.tokenizers[args.model_type].from_pretrained(self.args.pretrained_path,
                                                                           cache_dir="./cache/")
        with open(args.dic_path, 'r', encoding='utf-8') as fp:
            self._word2des = json.load(fp)

    def train(self):
        optimizer = transformers.AdamW(self.model.parameters(),
                                       lr=self.args.lr,  # args.learning_rate - default is 5e-5
                                       eps=self.args.eps  # args.adam_epsilon  - default is 1e-8
                                       )
        dataset = sampler.Dataset(self.sampler,
                                  tokenizer=self._tokenizer,
                                  word2des=self._word2des,
                                  padding_max=self.args.padding_max,
                                  margin_beta=self.args.margin_beta)
        data_loader = dataloader.DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=True)
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                                 num_warmup_steps=0,
                                                                 num_training_steps=len(data_loader) * self.args.epochs)
        self.model.cuda()
        loss_count = 0
        loss_max = 0
        for epoch in tqdm(range(self.args.epochs), desc="Training", total=self.args.epochs):
            dataset = sampler.Dataset(self.sampler,
                                      tokenizer=self._tokenizer,
                                      word2des=self._word2des,
                                      padding_max=self.args.padding_max,
                                      margin_beta=self.args.margin_beta)
            data_loader = dataloader.DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=True)
            loss_all = 0.0
            for batch in data_loader:
                optimizer.zero_grad()
                pos_output = self.model(input_ids=batch["pos_ids"].cuda(), token_type_ids=batch["pos_type_ids"].cuda(),
                                        attention_mask=batch["pos_attn_masks"].cuda())
                neg_output = self.model(input_ids=batch["neg_ids"].cuda(), token_type_ids=batch["neg_type_ids"].cuda(),
                                        attention_mask=batch["neg_attn_masks"].cuda())
                loss = self.model.margin_loss_fct(pos_output, neg_output, batch["margin"].cuda())
                loss.backward()
                optimizer.step()
                scheduler.step()
                loss_all += loss.item()
                self._log_tensorboard(self.args.log_label, "", loss.item(), loss_count)
                loss_count += 1
            loss_max = max(loss_max, loss_all)
            if epoch > 4 and loss_all > loss_max * 0.5:
                return False  # not converge
        return True

    def save_model(self):
        self.model.save_pretrained(self.args.save_path)
        self._tokenizer.save_pretrained(self.args.save_path)
