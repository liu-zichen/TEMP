import temp
import codecs
import torch
from transformers import BertTokenizer, ElectraTokenizer, AlbertTokenizer, RobertaTokenizer, XLNetTokenizer, XLMTokenizer
import json
from temp import TEMPBert, TEMPElectra, TEMPRoberta, TEMPAlbert, TEMPXLNet, TEMPXLM
from tqdm import tqdm


class Eval:
    models = {'bert': TEMPBert, 'electra': TEMPElectra, 'albert': TEMPAlbert, 'roberta': TEMPRoberta,
              'xlnet': TEMPXLNet, 'xlm': TEMPXLM}
    tokenizers = {'bert': BertTokenizer, 'electra': ElectraTokenizer, 'albert': AlbertTokenizer,
                  'roberta': RobertaTokenizer, 'xlnet': XLNetTokenizer, 'xlm': XLMTokenizer}

    def __init__(self, args):
        self.args = args
        with codecs.open(args.taxo_path, encoding='utf-8') as f:
            # TAXONOMY FILE FORMAT: hypernym <TAB> term
            tax_lines = f.readlines()
        tax_pairs = [line.strip().split("\t") for line in tax_lines]
        self.tax_graph = temp.TaxStruct(tax_pairs)
        with codecs.open(args.terms, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            self.terms = [line.strip() for line in lines]
        self.model = self.models[args.model_type].from_pretrained(args.model_path)
        self.sampler = temp.Sampler(self.tax_graph)

        with open(args.dic_path, 'r', encoding='utf-8') as fp:
            word2des = json.load(fp)
        tokenizer = self.tokenizers[args.model_type].from_pretrained(self.args.model_path)
        self._dataset = temp.Dataset(None, tokenizer, word2des, args.padding_max)  # 只用到了 encode

    def predict(self):
        eval_group, tags = self.gen_eval_data()
        results = []
        eval_max = 500
        self.model.cuda()
        for term in tqdm(self.terms, desc="Evaluating", total=len(self.terms)):
            data = eval_group[term]
            outputs = []
            data_l = int(data["ids"].size(0))
            for i in range(int((data_l - 1) / eval_max + 1)):
                begin = i * eval_max
                end = min((i + 1) * eval_max, data_l)
                with torch.no_grad():
                    output = self.model(input_ids=data["ids"][begin:end, ...].cuda(),
                                        token_type_ids=data["token_type_ids"][begin:end, ...].cuda(),
                                        attention_mask=data["attn_masks"][begin:end, ...].cuda())
                outputs.extend(output)
            outputs = torch.stack(outputs, dim=0)
            _, indices = outputs.squeeze().sort(descending=True)
            result = [tags[int(i)] for i in indices]
            results.append(result)
        return results

    def save_results(self, results):
        with codecs.open(self.args.output, mode='w+', encoding='utf-8') as fp:
            fp.write("\n".join(["\t".join(res) for res in results]))

    def gen_eval_data(self):
        path_group = dict()
        paths = []
        tags = []
        for k, v in self.tax_graph.node2path.items():
            tags.append(k)
            paths.append(v)
        for term in self.terms:
            ids_list = []
            token_type_ids_list = []
            attn_masks = []
            for path in paths:
                ids, token_type_ids, attn_mask = self._dataset.encode_path([term] + path)
                ids_list.append(ids)
                token_type_ids_list.append(token_type_ids)
                attn_masks.append(attn_mask)
            path_group[term] = dict(ids=torch.stack(ids_list, dim=0),
                                    token_type_ids=torch.stack(token_type_ids_list, dim=0),
                                    attn_masks=torch.stack(attn_masks, dim=0))
        return path_group, tags
