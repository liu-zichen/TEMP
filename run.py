import temp
import util
import torch
import scorer
import split_data
import configs


def run(split_args, train_args, eval_args):
    """
    This func includes three parts: dividing, training, evaluating.
    The data is transferred by files. So you can change code and do it step by step.
    """
    util.set_seed(configs.seed)
    split_args, train_args, eval_args = util.DotDict(split_args), util.DotDict(train_args), util.DotDict(eval_args)
    # divide raw_data to training and testing set; data will save in files
    split_data.spilt_data(split_args)
    # training
    trainer = temp.trainer.Trainer(train_args)
    trainer.train()
    trainer.save_model()
    # eval
    e = temp.Eval(eval_args)
    results = e.predict()
    with open(split_args.eval_path, encoding='utf-8') as fp:
        lines = fp.readlines()
    trues = [line.strip() for line in lines]
    acc, mrr, wu_p, wrong_set = scorer.score(results, trainer.tax_graph, trues)
    return acc, mrr, wu_p


if __name__ == '__main__':
    configs.seed = 0
    configs.model_type = "bert"
    dataset = "science"
    print(run(split_args=configs.split_configs[dataset],
              train_args=configs.train_configs[dataset],
              eval_args=configs.eval_configs[dataset]))
