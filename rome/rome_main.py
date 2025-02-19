from copy import deepcopy
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import nethook
from util.generate import generate_fast

from .compute_u import compute_u
from .compute_v import compute_v
from .rome_hparams import ROMEHyperParams

CONTEXT_TEMPLATES_CACHE = None


def apply_rome_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: List[Dict],
    hparams: ROMEHyperParams,
    copy=False,
    return_orig_weights=False,
    keep_original_weight=False,
    **kwargs,
) -> Tuple[AutoModelForCausalLM, List[str]]:
    """
    Returns a model with the desired changes.

    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.

    :return: (1) the updated model, (2) an original copy of the weights that changed
    """
    assert hparams.original_implementation or (
        hparams.enable_prompt_keys != hparams.enable_random_prefix_keys
    ), "Both prompt and random prefix keys are enabled"

    if not hparams.original_implementation:
        print(
            "Using key modification method:",
            "use_prompt_keys"
            if hparams.enable_prompt_keys
            else "use_random_prefix_keys",
        )

    # request = request[0]
    if copy:
        model = deepcopy(model)

    weights_copy = {}

    deltas = execute_rome(model, tok, request, hparams)

    with torch.no_grad():
    # with torch.autocast("cuda"): 
        for w_name, (delta_u, delta_v) in deltas.items():
            
            try:
                upd_matrix = delta_u.unsqueeze(1) @ delta_v.unsqueeze(0)
            except:
                delta_u = delta_u.half() 
                delta_v = delta_v.half()
                upd_matrix = delta_u.unsqueeze(1) @ delta_v.unsqueeze(0)

            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

            w[...] += upd_matrix

        print(f"New weights successfully inserted into {list(deltas.keys())}")

    if not keep_original_weight:
        weights_copy = {}

    return model, weights_copy


def execute_rome(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: Dict,
    hparams: ROMEHyperParams,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the ROME update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    # Update target and print info
    deltas = {}

    # Update target and print info
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"]["str"][0] != " ":
            # Space required for correct tokenization
            requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]
    for request in requests[:10]:
        print(
            f"MEMIT_ATTN request sample: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
        )

    # Retrieve weights that user desires to change
    weights = {
        f"{rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter( # transformer.h.{}.attn.out_proj
            model, f"{rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
        for rewrite_module_tmp in hparams.rewrite_module_tmps
    }
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}
    
    # Update loop: sequentially intervene at each specified layer
    deltas = {}
    for request in requests:
        for layer in sorted(hparams.layers):
            # Compute rank-1 update matrix
            left_vector: torch.Tensor = compute_u(
                model,
                tok,
                request,
                hparams,
                layer,
                get_context_templates(model, tok, hparams.context_template_length_params),
            )
            print("Left vector shape:", left_vector.shape)
            right_vector: torch.Tensor = compute_v(
                model,
                tok,
                request,
                hparams,
                layer,
                left_vector,
                get_context_templates(model, tok, hparams.context_template_length_params),
            )
            print("Right vector shape:", right_vector.shape)

            with torch.no_grad():
            # with torch.autocast("cuda"): 
                # Determine correct transposition of delta matrix
                weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
                try:
                    upd_matrix = left_vector.unsqueeze(1) @ right_vector.unsqueeze(0)
                    
                except:
                    left_vector = left_vector.half() 
                    right_vector = right_vector.half()
                    upd_matrix = left_vector.unsqueeze(1) @ right_vector.unsqueeze(0)
                upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

                # Update model weights and record desired changes in `delta` variable
                weights[weight_name][...] += upd_matrix
                deltas[weight_name] = (
                    left_vector.detach(),
                    right_vector.detach(),
                )

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by ROME does not match original weight shape. "
            "Check for bugs in the code?"
        )


def get_context_templates(model, tok, length_params):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = ["{}"] + [
            x.replace("{", "").replace("}", "") + ". {}"
            for x in sum(
                (
                    generate_fast(
                        model,
                        tok,
                        ["The", "Therefore", "Because", "I", "You"],
                        n_gen_per_prompt=n_gen // 5,
                        max_out_len=length,
                    )
                    for length, n_gen in length_params
                ),
                [],
            )
        ]

        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE
