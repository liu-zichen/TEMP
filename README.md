# TEMP
Code for EMNLP'21 Paper "[TEMP: Taxonomy Expansion with Dynamic Margin Loss through Taxonomy-Paths](https://aclanthology.org/2021.emnlp-main.313/)"

Our configs are in `configs.py`.
The program entry is `run.py`.

You can set your configs in code.

Folder Structure:
```text
files
├── data
│   ├── dic       // crawled dictionary (from wikipedia and PyDictionary)
│   └── raw_data  // SemEval 2016 Task 13 English dataset
├── temp  // main implement of TEMP model
├── configs.py // examples of configuration
├── scorer.py
└── run.py     // program entry
```


Please cite the paper if you found our method helpful. Thanks!

If you have any questions, please contact liuzichen@dbis.nankai.edu.cn.
