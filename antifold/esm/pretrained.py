# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
import urllib
import warnings
from argparse import Namespace
from pathlib import Path

import torch

import antifold.esm


def _has_regression_weights(model_name):
    """Return whether we expect / require regression weights;
    Right now that is all models except ESM-1v, ESM-IF, and partially trained ESM2 models
    """
    return not (
        "esm1v" in model_name
        or "esm_if" in model_name
        or "270K" in model_name
        or "500K" in model_name
    )


def load_model_and_alphabet(model_name):
    if model_name.endswith(".pt"):  # treat as filepath
        return load_model_and_alphabet_local(model_name)
    else:
        return load_model_and_alphabet_hub(model_name)


def load_hub_workaround(url):
    try:
        data = torch.hub.load_state_dict_from_url(
            url, progress=False, map_location="cpu"
        )
    except RuntimeError:
        # Pytorch version issue - see https://github.com/pytorch/pytorch/issues/43106
        fn = Path(url).name
        data = torch.load(
            f"{torch.hub.get_dir()}/checkpoints/{fn}",
            map_location="cpu",
        )
    except urllib.error.HTTPError as e:
        raise Exception(
            f"Could not load {url}, check if you specified a correct model name?"
        )
    return data


def load_regression_hub(model_name):
    url = f"https://dl.fbaipublicfiles.com/fair-esm/regression/{model_name}-contact-regression.pt"
    regression_data = load_hub_workaround(url)
    return regression_data


def _download_model_and_regression_data(model_name):
    url = f"https://dl.fbaipublicfiles.com/fair-esm/models/{model_name}.pt"
    model_data = load_hub_workaround(url)
    if _has_regression_weights(model_name):
        regression_data = load_regression_hub(model_name)
    else:
        regression_data = None
    return model_data, regression_data


def load_model_and_alphabet_hub(model_name):
    model_data, regression_data = _download_model_and_regression_data(model_name)
    return load_model_and_alphabet_core(model_name, model_data, regression_data)


def load_model_and_alphabet_local(model_location):
    """Load from local path. The regression weights need to be co-located"""
    model_location = Path(model_location)
    model_data = torch.load(str(model_location), map_location="cpu")
    model_name = model_location.stem
    if _has_regression_weights(model_name):
        regression_location = (
            str(model_location.with_suffix("")) + "-contact-regression.pt"
        )
        regression_data = torch.load(regression_location, map_location="cpu")
    else:
        regression_data = None
    return load_model_and_alphabet_core(model_name, model_data, regression_data)


def has_emb_layer_norm_before(model_state):
    """Determine whether layer norm needs to be applied before the encoder"""
    return any(
        k.startswith("emb_layer_norm_before") for k, param in model_state.items()
    )


def _load_model_and_alphabet_core_v1(model_data):
    import antifold.esm  # since esm.inverse_folding is imported below, you actually have to re-import esm here

    alphabet = antifold.esm.Alphabet.from_architecture(model_data["args"].arch)

    if "invariant_gvp" in model_data["args"].arch:
        import antifold.esm.inverse_folding

        model_type = antifold.esm.inverse_folding.gvp_transformer.GVPTransformerModel
        model_args = vars(model_data["args"])  # convert Namespace -> dict

        def update_name(s):
            # Map the module names in checkpoints trained with internal code to
            # the updated module names in open source code
            s = s.replace("W_v", "embed_graph.embed_node")
            s = s.replace("W_e", "embed_graph.embed_edge")
            s = s.replace("embed_scores.0", "embed_confidence")
            s = s.replace("embed_score.", "embed_graph.embed_confidence.")
            s = s.replace("seq_logits_projection.", "")
            s = s.replace("embed_ingraham_features", "embed_dihedrals")
            s = s.replace("embed_gvp_in_local_frame.0", "embed_gvp_output")
            s = s.replace("embed_features_in_local_frame.0", "embed_gvp_input_features")
            return s

        model_state = {
            update_name(sname): svalue
            for sname, svalue in model_data["model"].items()
            if "version" not in sname
        }

    else:
        raise ValueError("Unknown architecture selected")

    model = model_type(
        Namespace(**model_args),
        alphabet,
    )

    return model, alphabet, model_state


def _load_IF1_local():
    import antifold.esm  # since esm.inverse_folding is imported below, you actually have to re-import esm here

    alphabet = antifold.esm.Alphabet.from_architecture("vt_medium_with_invariant_gvp")

    import antifold.esm.inverse_folding

    model_type = antifold.esm.inverse_folding.gvp_transformer.GVPTransformerModel
    model_args = IF1_dict  # convert Namespace -> dict

    def update_name(s):
        # Map the module names in checkpoints trained with internal code to
        # the updated module names in open source code
        s = s.replace("W_v", "embed_graph.embed_node")
        s = s.replace("W_e", "embed_graph.embed_edge")
        s = s.replace("embed_scores.0", "embed_confidence")
        s = s.replace("embed_score.", "embed_graph.embed_confidence.")
        s = s.replace("seq_logits_projection.", "")
        s = s.replace("embed_ingraham_features", "embed_dihedrals")
        s = s.replace("embed_gvp_in_local_frame.0", "embed_gvp_output")
        s = s.replace("embed_features_in_local_frame.0", "embed_gvp_input_features")
        return s

    model = model_type(
        Namespace(**model_args),
        alphabet,
    )

    return model, alphabet


def load_model_and_alphabet_core(model_name, model_data, regression_data=None):
    if regression_data is not None:
        model_data["model"].update(regression_data["model"])

    model, alphabet, model_state = _load_model_and_alphabet_core_v1(model_data)

    expected_keys = set(model.state_dict().keys())
    found_keys = set(model_state.keys())

    if regression_data is None:
        expected_missing = {
            "contact_head.regression.weight",
            "contact_head.regression.bias",
        }
        error_msgs = []
        missing = (expected_keys - found_keys) - expected_missing
        if missing:
            error_msgs.append(f"Missing key(s) in state_dict: {missing}.")
        unexpected = found_keys - expected_keys
        if unexpected:
            error_msgs.append(f"Unexpected key(s) in state_dict: {unexpected}.")

        if error_msgs:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    model.__class__.__name__, "\n\t".join(error_msgs)
                )
            )
        if expected_missing - found_keys:
            warnings.warn(
                "Regression weights not found, predicting contacts will not produce correct results."
            )

    model.load_state_dict(model_state, strict=regression_data is not None)

    return model, alphabet


def esm_if1_gvp4_t16_142M_UR50():
    """Inverse folding model with 142M params, with 4 GVP-GNN layers, 8
    Transformer encoder layers, and 8 Transformer decoder layers, trained on
    CATH structures and 12 million alphafold2 predicted structures from UniRef50
    sequences.

    Returns a tuple of (Model, Alphabet).
    """
    return load_model_and_alphabet_hub("esm_if1_gvp4_t16_142M_UR50")


IF1_dict = {
'no_progress_bar': False,
'log_interval': 100,
'log_format': 'json',
'tensorboard_logdir': '',
'seed': 1238123,
'cpu': False,
'tpu': False,
'bf16': False,
'fp16': False,
'memory_efficient_bf16': False,
'memory_efficient_fp16': False,
'fp16_no_flatten_grads': False,
'fp16_init_scale': 128,
'fp16_scale_window': None,
'fp16_scale_tolerance': 0.0,
'min_loss_scale': 0.0001,
'threshold_loss_scale': None,
'user_dir': None,
'empty_cache_freq': 0,
'all_gather_list_size': 16384,
'model_parallel_size': 1,
'checkpoint_suffix': '',
'quantization_config_path': None,
'profile': False,
'criterion': 'inverse_folding_loss',
'tokenizer': None,
'bpe': None,
'optimizer': 'adam',
'lr_scheduler': 'inverse_sqrt',
'scoring': 'bleu',
'task': 'inverse_folding',
'num_workers': 8,
'skip_invalid_size_inputs_valid_test': False,
'max_tokens': 4096,
'max_sentences': None,
'required_batch_size_multiple': 8,
'dataset_impl': None,
'data_buffer_size': 10,
'train_subset': 'cath-chains-maxlen500/train,alphafold_relaxed_12M_flat/train',
'valid_subset': 'cath-chains-maxlen500/valid,cath-chains-maxlen500/short-valid,cath-chains-maxlen500/single-valid,alphafold_relaxed_12M_flat/valid',
'validate_interval': 1,
'validate_interval_updates': 0,
'validate_after_updates': 0,
'fixed_validation_seed': None,
'disable_validation': False,
'max_tokens_valid': 4096,
'max_sentences_valid': None,
'curriculum': 0,
'distributed_world_size': 32,
'distributed_rank': 0,
'distributed_backend': 'nccl',
'distributed_init_method': 'tcp://learnfair7689:14349',
'distributed_port': 14349,
'device_id': 0,
'distributed_no_spawn': False,
'ddp_backend': 'c10d',
'bucket_cap_mb': 25,
'fix_batches_to_gpus': False,
'find_unused_parameters': False,
'fast_stat_sync': False,
'broadcast_buffers': False,
'heartbeat_timeout': -1,
'distributed_wrapper': 'DDP',
'slowmo_momentum': None,
'slowmo_algorithm': 'LocalSGD',
'localsgd_frequency': 3,
'nprocs_per_node': 8,
'zero_sharding': 'none',
'arch': 'vt_medium_with_invariant_gvp',
'max_epoch': 200,
'max_update': 0,
'stop_time_hours': 0,
'clip_norm': 0.0,
'sentence_avg': False,
'update_freq': [1],
'lr': [0.001],
'min_lr': 1e-09,
'use_bmuf': False,
'save_dir': 'ablation3/ablation_transformer.seed0.s0.9.t0.0.cath+alphafold.ratios0.135.sub1.000.afns0.1.medium_with_invariant_gvp.bs4096.uf1.lr0.001.sqrtlr.warmup5000.mp0.15.geop0.05.upperspan30.ngpu32',
'restore_file': 'checkpoint_last.pt',
'finetune_from_model': None,
'reset_dataloader': False,
'reset_lr_scheduler': False,
'reset_meters': False,
'reset_optimizer': False,
'optimizer_overrides': '{}',
'save_interval': 1,
'save_interval_updates': 0,
'keep_interval_updates': -1,
'keep_last_epochs': 5,
'keep_best_checkpoints': -1,
'no_save': False,
'no_epoch_checkpoints': False,
'no_last_checkpoints': False,
'no_save_optimizer_state': False,
'best_checkpoint_metric': 'loss',
'maximize_best_checkpoint_metric': False,
'patience': -1,
'no_mid_epoch_validate': False,
'no_token_positional_embeddings': False,
'no_cross_attention': False,
'cross_self_attention': False,
'encoder_layerdrop': 0,
'decoder_layerdrop': 0,
'encoder_layers_to_keep': None,
'decoder_layers_to_keep': None,
'quant_noise_pq': 0,
'quant_noise_pq_block_size': 8,
'quant_noise_scalar': 0,
'joint_generation': False,
'et_gvp_as_feedforward': False,
'rescale_opt': 'token',
'use_inf_norm': False,
'adam_betas': '[0.9,0.999]',
'adam_eps': 1e-08,
'weight_decay': 0.0,
'use_old_adam': False,
'warmup_updates': 5000,
'warmup_init_lr': 1e-07,
'data': '/large_experiments/protein/data/inverse-folding/splits',
'binning_scheme': 'trRosetta_v2',
'include_angle': True,
'from_3d': True,
'in_memory_dataset': False,
'include_torsion': True,
'prepend_token': True,
'max_positions': 1024,
'train_sample_ratios': '1,0.135',
'valid_sample_ratios': '1,1,1,0.135',
'subsample_train': '1,1.000',
'train_crop_size': -1,
'min_num_source_residues': 0,
'partial_seq_masking': False,
'predict_next_only': False,
'test_time_coord_masking': False,
'source_coord_mask_prob': 0.15,
'source_coord_span_masking_span_upper': 30,
'source_coord_span_masking_span_lower': 1,
'source_coord_span_masking_geometric_p': 0.05,
'train_input_noises': '0,0.1',
'random_rotation': True,
'source_confidence_threshold': 0.9,
'target_confidence_threshold': 0.0,
'n_cond_score_buckets': 22,
'gvp_conditioning_encoder': True,
'gvp_conditioning_score_num_rbf': 16,
'gvp_node_input_dim_scalar': 7,
'gvp_node_input_dim_vector': 3,
'gvp_edge_input_dim_scalar': 34,
'gvp_edge_input_dim_vector': 1,
'gvp_vector_gate': True,
'gvp_attention_heads': 0,
'gvp_n_message_gvps': 3,
'gvp_n_edge_gvps_first_layer': 0,
'gvp_n_edge_gvps': 0,
'gvp_eps': 0.0001,
'gvp_layernorm': True,
'gvp_no_edge_orientation': False,
'gvp_conv_no_scalar_activation': False,
'gvp_conv_no_vector_activation': False,
'gvp_ignore_edges_without_coords': True,
'no_seed_provided': False,
'gvp_node_hidden_dim_scalar': 1024,
'gvp_node_hidden_dim_vector': 256,
'gvp_num_encoder_layers': 4,
'gvp_dropout': 0.1,
'gvp_top_k_neighbors': 30,
'gvp_edge_hidden_dim_scalar': 32,
'gvp_edge_hidden_dim_vector': 1,
'embed_gvp_in_global_frame': False,
'embed_gvp_in_local_frame': True,
'embed_ingraham_features': True,
'embed_rotation_frames': False,
'embed_rotation_quaternions': False,
'embed_features_in_global_frame': False,
'embed_features_in_local_frame': True,
'embed_scores': True,
'encoder_embed_dim': 512,
'encoder_ffn_embed_dim': 2048,
'encoder_attention_heads': 8,
'encoder_layers': 8,
'decoder_embed_dim': 512,
'decoder_ffn_embed_dim': 2048,
'decoder_attention_heads': 8,
'decoder_layers': 8,
'attention_dropout': 0.1,
'encoder_normalize_before': True,
'decoder_normalize_before': True,
'pretrained_gvp_stem': None,
'frozen_gvp_stem': False,
'pretrained_decoder': None,
'frozen_decoder': False,
'encoder_edge_attn_layers': 0,
'edge_attn_distance_cutoff': -1,
'edge_embed_dim': 0,
'embed_edge_vectors': False,
'distance_noise': 0.0,
'gvp_distance_noise': 0.0,
'equivariant_attn_layers': 0,
'embed_patch_layers': 0,
'encoder_embed_path': None,
'encoder_learned_pos': False,
'decoder_embed_path': None,
'decoder_learned_pos': False,
'activation_dropout': 0.0,
'activation_fn': 'relu',
'dropout': 0.1,
'adaptive_softmax_cutoff': None,
'adaptive_softmax_dropout': 0,
'share_decoder_input_output_embed': False,
'share_all_embeddings': False,
'adaptive_input': False,
'decoder_output_dim': 512,
'decoder_input_dim': 512,
'no_scale_embedding': False,
'layernorm_embedding': False,
'tie_adaptive_weights': False,
'max_source_positions': 1024,
'max_target_positions': 1024
}