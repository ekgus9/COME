# CoME

This repository is the official implementation of the paper "CoME: An Unlearning-based Approach to Conflict-free Model Editing", NAACL 2025.

## Abstract
Large language models (LLMs) often retain outdated or incorrect information from pretraining, which undermines their reliability. While model editing methods have been developed to address such errors without full retraining, they frequently suffer from knowledge conflicts, where outdated information interferes with new knowledge. In this work, we propose Conflict-free Model Editing (CoME), a novel framework that enhances the accuracy of knowledge updates in LLMs by selectively removing outdated knowledge. CoME leverages unlearning to mitigate knowledge interference, allowing new information to be integrated without compromising relevant linguistic features. Through experiments on GPT-J and LLaMA3 using Counterfact and ZsRE datasets, we demonstrate that CoME improves both editing accuracy and model reliability when applied to existing editing methods. Our results highlight that the targeted removal of outdated knowledge is crucial for enhancing model editing effectiveness and maintaining the model’s generative performance

## Requirements

```
conda create -n come python=3.10
pip install -r requirements.txt
```

## Quick Start
### 1. Edit GPT-J (6B) model on Counterfact dataset using CoME

```
python evaluate.py --alg_name COME_MEMIT --model_name EleutherAI/gpt-j-6B --hparams_fname EleutherAI_gpt-j-6B.json --ds_name mcf --num_edits 1
```

### 2. Summarize the results

```
python summarize.py --dir_name=COME_MEMIT --runs=run_<run_code>
```

Our code is based on  [``PMET``](https://github.com/xpq-tech/PMET).

## Citation

```
@misc{jung2025comeunlearningbasedapproachconflictfree,
      title={CoME: An Unlearning-based Approach to Conflict-free Model Editing}, 
      author={Dahyun Jung and Jaehyung Seo and Jaewook Lee and Chanjun Park and Heuiseok Lim},
      year={2025},
      eprint={2502.15826},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.15826}, 
}
```
