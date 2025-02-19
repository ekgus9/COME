"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_zsre` with the
appropriate arguments, which returns a dictionary containing them.
"""

import typing
from itertools import chain

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from dsets import AttributeSnippets

def compute_acc(model_name,tok, rewrite_prompts, target_new, model):
    try:
        tn = target_new["str"]
    except:
        tn = target_new

    # if 'llama' in model_name:
    #     prefix_lens = [len(n)-1 for n in tok(rewrite_prompts)["input_ids"]]
    # else:
    prefix_lens = [len(n) for n in tok(rewrite_prompts)["input_ids"]]

    prompt_tok = tok(
        [
            f"{prefix} {suffix}"
            for prefix in rewrite_prompts
            for suffix in [tn]
        ],
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    a_tok = tok(f" {tn}", add_special_tokens=False)["input_ids"]
    choice_a_len = len(a_tok)

    with torch.no_grad():
        try:
            logits = model(**prompt_tok).logits
        except TypeError:
            prompt_tok = {'input_ids': prompt_tok['input_ids'], 'attention_mask': prompt_tok['attention_mask']}
            logits = model(**prompt_tok).logits

    targets_correct = []

    for i in range(logits.size(0)):
        cur_len = choice_a_len 

        
        for sw in range(cur_len):
                correct = True
                cur_tok = a_tok

                if logits[i, prefix_lens[i // 2] + sw - 1, :].argmax().item() != cur_tok[sw]:
                        
                        correct = False
                        # break
                targets_correct.append(correct)

    return targets_correct

def compute_rewrite_quality_zsre(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: typing.Dict,
    snips: AttributeSnippets,
    vec: TfidfVectorizer,
    model_name = None
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """

        # First, unpack rewrite evaluation record.
    subject, target_new, target_true = (
            record["requested_rewrite"][x] for x in ["subject", "target_new", "target_true"]
        )
    rewrite_prompts = [record["requested_rewrite"]["prompt"].format(subject)]
    paraphrase_prompts = record["paraphrase_prompts"]
    neighborhood_prompts = record["neighborhood_prompts"]

        # Form a list of lists of prefixes to test.
    prob_prompts = [
            rewrite_prompts,
            paraphrase_prompts,
        ]
        # Flatten all the evaluated prefixes into one list.
    target_tok = tok(" " + target_new["str"], add_special_tokens=False)["input_ids"]

    if 'gpt' in model_name:

        inp_prompts_og = list(chain(*prob_prompts))
        if 'llama' in model_name:
            inp_prompts = [
                el + tok.decode(target_tok[:i], add_special_tokens=False)
                for el in inp_prompts_og
                for i in range(len(target_tok))
            ]
        else:
            inp_prompts = [
                el + tok.decode(target_tok[:i], add_special_tokens=False)
                for el in inp_prompts_og
                for i in range(len(target_tok))
            ]
        inp_targets = [
            tok.decode(target_tok[i], add_special_tokens=False)
            for _ in range(len(inp_prompts_og))
            for i in range(len(target_tok))
        ]

        stuff_probs = test_batch_prediction_acc(model, tok, inp_prompts, inp_targets)

        # # Predict for neighborhood prompts (dictionary format).
        neighborhood_correct = test_batch_prediction_acc(
            model,
            tok,
            [
                el["prompt"].format(record["requested_rewrite"])
                for el in neighborhood_prompts
            ],
            [el["target"] for el in neighborhood_prompts],
        )

        probs = stuff_probs + neighborhood_correct

        # Unflatten the results again into a list of lists.
        cutoffs = [0] + np.cumsum(
            [l * len(target_tok) for l in map(len, prob_prompts)]
        ).tolist()
        ret_probs = [probs[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))]
        # Structure the restuls as a dictionary.
        ret = {
            f"{key}_correct": ret_probs[i]
            for i, key in enumerate(
                [
                    "rewrite_prompts",
                    "paraphrase_prompts",
                ]
            )
        }

        ret["neighborhood_prompts_correct"] = neighborhood_correct

    ##########
    else:
        rewrite_prompts_correct = compute_acc(model_name, tok, rewrite_prompts, target_new, model)
        paraphrase_prompts_correct = compute_acc(model_name, tok, paraphrase_prompts, target_new, model)

        neighborhood_prompts_com = [neighborhood_prompts["prompt"].format(record["requested_rewrite"])]
        target_neighborhood_com = neighborhood_prompts["target"]

        neighborhood_prompts_correct = compute_acc(model_name, tok, neighborhood_prompts_com, target_neighborhood_com, model)

        ret = {}

        ret["rewrite_prompts_correct"] = rewrite_prompts_correct
        ret["paraphrase_prompts_correct"] = paraphrase_prompts_correct
        ret["neighborhood_prompts_correct"] = neighborhood_prompts_correct

    #########

    return ret


def test_batch_prediction_acc(model, tok, prompts: typing.List[str], target):
    prompt_tok = tok(
        prompts,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        logits = model(**prompt_tok).logits
        last_non_masked = prompt_tok["attention_mask"].sum(1) - 1
        to_gather = last_non_masked.unsqueeze(1).repeat(1, logits.size(-1)).unsqueeze(1)
        gathered = torch.gather(logits, 1, to_gather).squeeze(1)
        ans = torch.argmax(gathered, dim=1)

        correct_id = tok(target, padding=True, return_tensors="pt", add_special_tokens=False).to("cuda")[
            "input_ids"
        ]
        # Temporary hack to deal with foreign characters.
        correct_id = correct_id[:, 0].squeeze()

        return (ans == correct_id).detach().cpu().numpy().tolist()
