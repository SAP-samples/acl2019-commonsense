# Attention Is (not) All You Need for Commonsense Reasoning
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![REUSE status](https://api.reuse.software/badge/github.com/SAP-samples/acl2019-commonsense)](https://api.reuse.software/info/github.com/SAP-samples/acl2019-commonsense)


#### News
- Updated to version 0.2.0: added stand-alone script and example script
- Updated to version 0.1.0: allows replication of our ACL'19 paper results


## Description:
![Schematic Illustration MAS](https://github.com/SAP-samples/acl2019-commonsense/blob/master/img/mas_illustration.png)
The recently introduced [BERT (Deep Bidirectional Transformers for Language Understanding)](https://github.com/google-research/bert) [1] model exhibits strong performance on several language understanding benchmarks. In this work, we describe a simple re-implementation of BERT for commonsense reasoning. We show that the attentions produced by BERT can be directly utilized for tasks such as the Pronoun Disambiguation Problem (PDP) and Winograd Schema Challenge (WSC). Our proposed attention-guided commonsense reasoning method is conceptually simple yet empirically powerful. Experimental analysis on multiple datasets demonstrates that our proposed system performs remarkably well on all cases while outperforming the previously reported state of the art by a margin. While results suggest that BERT seems to implicitly learn to establish complex relationships between entities, solving commonsense reasoning tasks might require more than unsupervised models learned from huge text corpora.
The sample code provided within this repository allows to replicate the results reported in the paper for PDP and WSC.
#### Authors:
 - [Tassilo Klein](https://tjklein.github.io/)
 - [Moin Nabi](https://moinnabi.github.io/)

## Requirements
- [Python](https://www.python.org/)
- [PyTorch](https://pytorch.org/)
- [Huggingface Tranformers](https://github.com/huggingface/transformers)
- [BertViz](https://github.com/jessevig/bertviz)
- [WSC,PDP data](https://cs.nyu.edu/faculty/davise/papers/WinogradSchemas/)

## Download and Installation
1. Install BertViz by cloning the repository and getting dependencies:
```
git clone https://github.com/jessevig/bertviz.git
cd bertviz
pip install -r requirements.txt
cd ..
```

2. To replicate the results proceed to step 3). If you want to run the stand-alone version, you can just use [MAS.py](https://github.com/SAP-samples/acl2019-commonsense-reasoning/blob/master/MAS.py). Usage is showcased in the Jupyter Notebook example [MAS_Example.ipynb](https://github.com/SAP-samples/acl2019-commonsense-reasoning/blob/master/MAS_Example.ipynb).

3. Add BertViz path to Python path:
```
  export PYTHONPATH=$PYTHONPATH:/home/ubuntu/bertviz/
```
alternatively, you can add the statement to commonsense.py after importing of sys, e.g.
```
sys.path.append("/home/ubuntu/bertviz/")
```

4. Clone this repository and install dependencies:
```
git clone https://github.com/SAP/acl2019-commonsense-reasoning
cd acl2019-commonsense-reasoning
pip install -r requirements.txt
```

5. Create 'data' sub-directory and download files for PDP and WSC challenges:
```
mkdir data
wget https://cs.nyu.edu/faculty/davise/papers/WinogradSchemas/PDPChallenge2016.xml
wget https://cs.nyu.edu/faculty/davise/papers/WinogradSchemas/WSCollection.xml
cd ..
```
6. Run the scripts from the paper

For replicating the results on WSC:
```
python commonsense.py --data_dir=~/acl2019-commonsense-reasoning/data/ --bert_mode=bert-base-uncased --do_lower_case --task_name=MNLI
```

For replicating the results on PDP:
```
python commonsense.py --data_dir=~/acl2019-commonsense-reasoning/data/ --bert_mode=bert-base-uncased --do_lower_case --task_name=pdp
```

For more information on the individual functions, please refer to their doc strings.

## Related Work
See our latest work accepted at [ACL'20](http://acl2020.org/) on commonsense reasoning using contrastive self-supervised learning. [arXiv](https://arxiv.org/abs/2005.00669), [GitHub](https://github.com/SAP-samples/acl2020-commonsense/)

## Known Issues
No issues known


## How to obtain support
This project is provided "as-is" and any bug reports are not guaranteed to be fixed.


## Citations
If you use this code in your research,
please cite:

```
@inproceedings{klein-nabi-2019-attention,
    title = "Attention Is (not) All You Need for Commonsense Reasoning",
    author = "Klein, Tassilo  and
      Nabi, Moin",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1477",
    doi = "10.18653/v1/P19-1477",
    pages = "4831--4836",
    abstract = "The recently introduced BERT model exhibits strong performance on several language understanding benchmarks. In this paper, we describe a simple re-implementation of BERT for commonsense reasoning. We show that the attentions produced by BERT can be directly utilized for tasks such as the Pronoun Disambiguation Problem and Winograd Schema Challenge. Our proposed attention-guided commonsense reasoning method is conceptually simple yet empirically powerful. Experimental analysis on multiple datasets demonstrates that our proposed system performs remarkably well on all cases while outperforming the previously reported state of the art by a margin. While results suggest that BERT seems to implicitly learn to establish complex relationships between entities, solving commonsense reasoning tasks might require more than unsupervised models learned from huge text corpora.",
}
```

## References
- [1] J. Devlin et al., BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, 2018, https://arxiv.org/abs/1810.04805.


## License
Copyright (c) 2019-2020 SAP SE or an SAP affiliate company. All rights reserved. This project is licensed under the Apache Software License, version 2.0 except as noted otherwise in the [LICENSE](LICENSES/Apache-2.0.txt).
