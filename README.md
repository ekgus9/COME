# CoME

This repository is the official implementation of the paper "CoME: An Unlearning-based Approach to Conflict-free Model Editing", NAACL 2025.

## Requirements

    ``` bash
    conda create -n come python=3.10
    pip install -r requirements.txt
    ```

## Quick Start
### 1. Edit GPT-J (6B) model on Counterfact dataset using CoME

    ```
    python evaluate.py --alg_name COME_MEMIT --model_name EleutherAI/gpt-j-6B --hparams_fname EleutherAI_gpt-j-6B.json --ds_name mcf --num_edits 1 --dataset_size_limit 10
    ```

### 2. Summarize the results

    ```
    python summarize.py --dir_name=COME_MEMIT --runs=run_<run_code>
    ```

Our code is based on  [``PMET``](https://github.com/xpq-tech/PMET).

## Citation

```

```