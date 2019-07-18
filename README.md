# Attention Is (not) All You Need for Commonsense Reasoning

## Description:
The recently introduced [BERT (Deep Bidirectional Transformers for Language Understanding)](https://github.com/google-research/bert) [1] model exhibits strong performance on several language understanding benchmarks. In this work, we describe a simple re-implementation of BERT for commonsense reasoning. We show that the attentions produced by BERT can be directly utilized for tasks such as the Pronoun Disambiguation Problem (PDP) and Winograd Schema Challenge (WSC). Our proposed attention-guided commonsense reasoning method is conceptually simple yet empirically powerful. Experimental analysis on multiple datasets demonstrates that our proposed system performs remarkably well on all cases while outperforming the previously reported state of the art by a margin. While results suggest that BERT seems to implicitly learn to establish complex relationships between entities, solving commonsense reasoning tasks might require more than unsupervised models learned from huge text corpora.
The sample code provided within this repository allows to replicate the results reported in the paper for PDP and WSC.

## Requirements
- [Python](https://www.python.org/)
- [PyTorch](https://pytorch.org/)
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

## Known Issues
No issues known


## How to obtain support
This project is provided "as-is" and any bug reports are not guaranteed to be fixed.


## Citations
If you use this code in your research,
please cite:

```
@article{DBLP:journals/corr/abs-1905-13497,
  author    = {Tassilo Klein and
               Moin Nabi},
  title     = {Attention Is (not) All You Need for Commonsense Reasoning},
  journal   = {CoRR},
  volume    = {abs/1905.13497},
  year      = {2019},
  url       = {http://arxiv.org/abs/1905.13497},
  archivePrefix = {arXiv},
  eprint    = {1905.13497},
  timestamp = {Mon, 03 Jun 2019 13:42:33 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1905-13497},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## References
- [1] J. Devlin et al., BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, 2018, https://arxiv.org/abs/1810.04805.


## License

This project is licensed under SAP Sample Code License Agreement except as noted otherwise in the [LICENSE file](LICENSE.md).
