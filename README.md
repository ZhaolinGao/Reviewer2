# Reviewer2

This repo covers the implementation for our paper [Reviewer2](https://arxiv.org/pdf/2402.10886.pdf). 

Zhaolin Gao, Kianté Brantley, and Thorsten Joachims. "Reviewer2: Optimizing Review Generation Through Prompt Generation" 

## Table of Contents 

* [Dataset](#dataset)
* [Model](#model)
* [Environment](#environment)
* [Demo](#demo)
* [Cite](#cite)
* [Acknowledgments](#acknowledgments)

## Dataset

The dataset created using our PGE pipeline is uploaded to huggingface. We incorporate parts of the [PeerRead](https://github.com/allenai/PeerRead) and [NLPeer](https://github.com/UKPLab/nlpeer) datasets along with an update-to-date crawl from ICLR and NeurIPS on [OpenReview](https://openreview.net/) and [NeurIPS Proceedings](http://papers.neurips.cc/).

[Raw data](https://huggingface.co/datasets/GitBag/Reviewer2_PGE_raw) contains paper contents (json and/or PDF), review contents (json), and metadata (json).
[Cleaned data](https://huggingface.co/datasets/GitBag/Reviewer2_PGE_cleaned) contains our split for train, validation, and test and can be directly used.

## Model

Prompt generation model [Mp](https://huggingface.co/GitBag/Reviewer2_Mp).

Review generation model [Mr](https://huggingface.co/GitBag/Reviewer2_Mr).

## Environment

The demo requires java 11.0.20 and the following python environment:
```
python 3.9.18
pytorch 2.1.0
transformers 4.34.0
flash-attn 2.3.2
einops 0.7.0
```

## Demo

This demo generates a review for a given PDF file.

#### 1. Convert PDF to json file:
```
java -Xmx6g -jar ./science-parse-cli-assembly-2.0.3.jar PATH_TO_PDF -o JSON_SAVE_DIRECTORY
```

#### 2. Generate the review:
```
python demo.py --json_path PATH_TO_JSON
```

## Cite
Please cite our paper if you use this implementation in your own work:
```
@misc{gao2024reviewer2,
      title={Reviewer2: Optimizing Review Generation Through Prompt Generation}, 
      author={Zhaolin Gao and Kianté Brantley and Thorsten Joachims},
      year={2024},
      eprint={2402.10886},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Acknowledgments
Thanks for [Longformer](https://github.com/allenai/longformer) and [Science Parse](https://github.com/allenai/science-parse), on which this repository is initially based.
