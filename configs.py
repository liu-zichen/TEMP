# dataset = "science"
seed = 0
model_type2path = {
    "electra": "google/electra-base-discriminator",
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
    "albert": "albert-base-v2",
    # "xlm": "",
    # "xlnet": "xlnet-base-cased"
}
model_type = "bert"

# configs for divide train/test set
split_configs = {
    "science": dict(
        taxo_path="./data/raw_data/science_wordnet_en.taxo",
        train_path="./data/train/science_train.taxo",
        terms_path="./data/eval/science_eval.terms",
        eval_path="./data/eval/science_eval.taxo"
    ),
    "food": dict(
        taxo_path="./data/raw_data/food_wordnet_en.taxo",
        train_path="./data/train/food_train.taxo",
        terms_path="./data/eval/food_eval.terms",
        eval_path="./data/eval/food_eval.taxo"
    ),
    "environment": dict(
        taxo_path="./data/raw_data/environment_eurovoc_en.taxo",
        train_path="./data/train/environment_train.taxo",
        terms_path="./data/eval/environment_eval.terms",
        eval_path="./data/eval/environment_eval.taxo"
    )
}

# configs for training
train_configs = {
    "science": dict(
        # paths
        taxo_path="./data/train/science_train.taxo",
        pretrained_path=model_type2path[model_type],
        dic_path="./data/dic/dic.json",
        save_path="./data/models/trained_science_" + model_type + "/",
        log_path="./data/log/",
        # config
        seed=0,
        margin_beta=0.2,
        epochs=110,
        batch_size=32,
        lr=2e-5,
        eps=1e-8,
        padding_max=110,
        log_label="science",
        model_type=model_type
    ),
    "food": dict(
        # paths
        taxo_path="./data/train/food_train.taxo",
        pretrained_path=model_type2path[model_type],
        dic_path="./data/dic/dic.json",
        save_path="./data/models/trained_food_" + model_type + "/",
        log_path="./data/log/",
        # config
        seed=seed,
        margin_beta=0.2,
        epochs=300,
        batch_size=32,
        lr=1e-5,
        eps=1e-8,
        padding_max=150,
        log_label="food",
        model_type=model_type
    ),
    "environment": dict(
        # paths
        taxo_path="./data/train/environment_train.taxo",
        pretrained_path=model_type2path[model_type],
        dic_path="./data/dic/dic.json",
        save_path="./data/models/trained_environment_" + model_type + "/",
        log_path="./data/log/",
        # config
        seed=seed,
        margin_beta=0.2,
        epochs=110,
        batch_size=32,
        lr=2e-5,
        eps=1e-8,
        padding_max=110,
        log_label="environment",
        model_type=model_type
    )
}

eval_configs = {
    "science": dict(
        # paths
        taxo_path="./data/train/science_train.taxo",
        model_path="./data/models/trained_science_" + model_type + "/",
        dic_path="./data/dic/dic.json",
        terms="./data/eval/science_eval.terms",
        output="./data/result/science.results",
        # config
        padding_max=150,
        model_type=model_type
    ),
    "food": dict(
        # paths
        taxo_path="./data/train/food_train.taxo",
        model_path="./data/models/trained_food_" + model_type + "/",
        dic_path="./data/dic/dic.json",
        terms="./data/eval/food_eval.terms",
        output="./data/result/food.results",
        # config
        padding_max=150,
        model_type=model_type
    ),
    "environment": dict(
        # paths
        taxo_path="./data/train/food_train.taxo",
        model_path="./data/models/trained_food_" + model_type + "/",
        dic_path="./data/dic/dic.json",
        terms="./data/eval/food_eval.terms",
        output="./data/result/food.results",
        # config
        padding_max=150,
        model_type=model_type
    )
}
