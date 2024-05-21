# https://github.com/facebookresearch/esm/blob/main/esm/inverse_folding/multichain_util.py
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import List

import biotite.structure
import numpy as np
# import torch
from biotite.sequence import ProteinSequence
# from biotite.structure.residues import get_residues
from biotite.structure.residues import get_residue_starts

from antifold.esm.inverse_folding.util import (get_encoder_output,
                                               get_sequence_loss,
                                               load_structure)


def get_atom_coords_residuewise(atoms: List[str], struct: biotite.structure.AtomArray):
    """
    Example for atoms argument: ["N", "CA", "C"]
    """

    def filterfn(s, axis=None):
        filters = np.stack([s.atom_name == name for name in atoms], axis=1)
        sum = filters.sum(0)
        if not np.all(sum <= np.ones(filters.shape[1])):
            raise RuntimeError("structure has multiple atoms with same name")
        index = filters.argmax(0)
        coords = s[index].coord
        coords[sum == 0] = float("nan")
        return coords

    return biotite.structure.apply_residue_wise(struct, struct, filterfn)


def get_residues_imgt(array):
    """Get the residue IDs+insertion code (IMGT) and names of an atom array (stack)"""
    starts = get_residue_starts(array)

    # Position + insertion code
    res_pos = array.res_id[starts]
    res_ins = array.ins_code[starts]
    res_chains = array.chain_id[starts]
    res_posinschain = np.array(
        list(
            [
                f"{pos}{ins}{chain}"
                for pos, ins, chain in zip(res_pos, res_ins, res_chains)
            ]
        )
    )

    # Residue name, 3-letter code
    res_name = array.res_name[starts]

    return res_pos, res_posinschain, res_name


def extract_coords_from_structure(structure: biotite.structure.AtomArray):
    """
    Args:
        structure: An instance of biotite AtomArray
    Returns:
        Tuple (coords, seq, res_posinschain)
            - coords (array) is L x 3 x 3 for N, CA, C coordinates
            - seq (str) is the extracted sequence
            - res_posinschain (str) is the residue position IDs with insertion code (e.g. IMGT numbering)
    """
    coords = get_atom_coords_residuewise(["N", "CA", "C"], structure)
    res_pos, res_posinschain, res_3letter = get_residues_imgt(structure)
    seq = "".join([ProteinSequence.convert_letter_3to1(r) for r in res_3letter])
    return coords, seq, res_pos, res_posinschain


def extract_coords_from_complex(structure: biotite.structure.AtomArray):
    """
    Args:
        structure: biotite AtomArray
    Returns:
        Tuple (coords_list, seq_list)
        - coords: Dictionary mapping chain ids to L x 3 x 3 array for N, CA, C
          coordinates representing the backbone of each chain
        - seqs: Dictionary mapping chain ids to native sequences of each chain
    """
    coords = {}
    seqs = {}
    positions = {}
    positions_ins = {}
    all_chains = biotite.structure.get_chains(structure)
    for chain_id in all_chains:
        chain = structure[structure.chain_id == chain_id]
        (
            coords[chain_id],
            seqs[chain_id],
            positions[chain_id],
            positions_ins[chain_id],
        ) = extract_coords_from_structure(chain)
    return coords, seqs, positions, positions_ins


def load_complex_coords(fpath, chains):
    """
    Args:
        fpath: filepath to either pdb or cif file
        chains: the chain ids (the order matters for autoregressive model)
    Returns:
        Tuple (coords_list, seq_list)
        - coords: Dictionary mapping chain ids to L x 3 x 3 array for N, CA, C
          coordinates representing the backbone of each chain
        - seqs: Dictionary mapping chain ids to native sequences of each chain
    """
    assert os.path.exists(fpath)

    structure = load_structure(fpath, chains)
    return extract_coords_from_complex(structure)


def _concatenate_coords(coords, target_chain_id, padding_length=10):
    """
    Args:
        coords: Dictionary mapping chain ids to L x 3 x 3 array for N, CA, C
            coordinates representing the backbone of each chain
        target_chain_id: The chain id to sample sequences for
        padding_length: Length of padding between concatenated chains
    Returns:
        Tuple (coords, seq)
            - coords is an L x 3 x 3 array for N, CA, C coordinates, a
              concatenation of the chains with padding in between
            - seq is the extracted sequence, with padding tokens inserted
              between the concatenated chains
    """
    pad_coords = np.full((padding_length, 3, 3), np.nan, dtype=np.float32)
    # For best performance, put the target chain first in concatenation.
    coords_list = [coords[target_chain_id]]
    for chain_id in coords:
        if chain_id == target_chain_id:
            continue
        coords_list.append(pad_coords)
        coords_list.append(coords[chain_id])
    coords_concatenated = np.concatenate(coords_list, axis=0)
    return coords_concatenated


def concatenate_coords_HL(
    coords_dict, seq_dict, pos_dict, posins_dict, heavy_chain_id, padding_length=10
):
    """
    Args:
        coords: Dictionary mapping chain ids to L x 3 x 3 array for N, CA, C
            coordinates representing the backbone of each chain
        seq_dict: Dictionary mapping chain ids to native sequences of each chain
        pos_dict: Residue position IDs (e.g. IMGT numbering)
        heavy_chain_id: Heavy chain id, always start
        padding_length: Length of padding between concatenated chains
    Returns:
        Tuple (coords, seq, pos)
            - coords_concatenated is an L x 3 x 3 array for N, CA, C coordinates, a
              concatenation of the chains with padding in between
            - seq_str is the extracted sequence, with padding tokens inserted
              between the concatenated chains
            - pos_concatenated is the residue position IDs, with NaNs inserted
    """

    pad_coords = np.full((padding_length, 3, 3), np.inf, dtype=np.float32)
    pad_pos = np.full((padding_length), np.nan, dtype=np.float32)
    pad_seq = "-" * padding_length

    # Make sure always only 2 chain (H and L)
    assert len(coords_dict) == 2

    # Add heavy chain
    coords_list = [coords_dict[heavy_chain_id]]
    pos_list = [pos_dict[heavy_chain_id]]
    # Position + insertion (string) # MH
    posins_list = [posins_dict[heavy_chain_id]]
    seq_str = seq_dict[heavy_chain_id]

    # Add light chain, with 10x padding between H and L chains
    for chain_id in coords_dict:
        if chain_id == heavy_chain_id:
            continue

        coords_list.append(pad_coords)
        coords_list.append(coords_dict[chain_id])
        pos_list.append(pad_pos)
        pos_list.append(pos_dict[chain_id])
        # Position + insertion (string) # MH
        posins_list.append(pad_pos)
        posins_list.append(posins_dict[chain_id])
        seq_str += pad_seq
        seq_str += seq_dict[chain_id]

    coords_concatenated = np.concatenate(coords_list, axis=0)
    pos_concatenated = np.concatenate(pos_list, axis=0)
    # Position + insertion (string) # MH
    posins_concatenated = np.concatenate(posins_list, axis=0)

    if not len(seq_str) == len(coords_concatenated) == len(pos_concatenated):
        print(
            f"Length mismatch: seq_str {len(seq_str)} coords_concatenated {len(coords_concatenated)} pos_concatenated {len(pos_concatenated)}"
        )
        raise ValueError

    return coords_concatenated, seq_str, pos_concatenated, posins_concatenated


def concatenate_coords_any(
    coords_dict, seq_dict, pos_dict, posins_dict, chains, padding_length=10
):
    """
    Args:
        coords: Dictionary mapping chain ids to L x 3 x 3 array for N, CA, C
            coordinates representing the backbone of each chain
        seq_dict: Dictionary mapping chain ids to native sequences of each chain
        pos_dict: Residue position IDs (e.g. IMGT numbering)
        first_chain_id: First chain id, always start
        padding_length: Length of padding between concatenated chains
    Returns:
        Tuple (coords, seq, pos)
            - coords_concatenated is an L x 3 x 3 array for N, CA, C coordinates, a
              concatenation of the chains with padding in between
            - seq_str is the extracted sequence, with padding tokens inserted
              between the concatenated chains
            - pos_concatenated is the residue position IDs, with NaNs inserted
    """

    pad_coords = np.full((padding_length, 3, 3), np.inf, dtype=np.float32)
    pad_pos = np.full((padding_length), np.nan, dtype=np.float32)
    pad_seq = "-" * padding_length

    # Add heavy chain
    first_chain_id = chains[0]
    coords_list = [coords_dict[first_chain_id]]
    pos_list = [pos_dict[first_chain_id]]
    # Position + insertion (string) # MH
    posins_list = [posins_dict[first_chain_id]]
    seq_str = seq_dict[first_chain_id]

    # Add remaining chains, with 10x padding between H and L chains
    for chain_id in chains:
        if chain_id == first_chain_id:
            continue

        coords_list.append(pad_coords)
        coords_list.append(coords_dict[chain_id])
        pos_list.append(pad_pos)
        pos_list.append(pos_dict[chain_id])
        # Position + insertion (string) # MH
        posins_list.append(pad_pos)
        posins_list.append(posins_dict[chain_id])
        seq_str += pad_seq
        seq_str += seq_dict[chain_id]

    coords_concatenated = np.concatenate(coords_list, axis=0)
    pos_concatenated = np.concatenate(pos_list, axis=0)
    # Position + insertion (string) # MH
    posins_concatenated = np.concatenate(posins_list, axis=0)

    if not len(seq_str) == len(coords_concatenated) == len(pos_concatenated):
        print(
            f"Length mismatch: seq_str {len(seq_str)} coords_concatenated {len(coords_concatenated)} pos_concatenated {len(pos_concatenated)}"
        )
        raise ValueError

    return coords_concatenated, seq_str, pos_concatenated, posins_concatenated


def sample_sequence_in_complex(
    model, coords, target_chain_id, temperature=1.0, padding_length=10
):
    """
    Samples sequence for one chain in a complex.
    Args:
        model: An instance of the GVPTransformer model
        coords: Dictionary mapping chain ids to L x 3 x 3 array for N, CA, C
            coordinates representing the backbone of each chain
        target_chain_id: The chain id to sample sequences for
        padding_length: padding length in between chains
    Returns:
        Sampled sequence for the target chain
    """
    target_chain_len = coords[target_chain_id].shape[0]
    all_coords = _concatenate_coords(coords, target_chain_id)

    # Supply padding tokens for other chains to avoid unused sampling for speed
    padding_pattern = ["<pad>"] * all_coords.shape[0]
    for i in range(target_chain_len):
        padding_pattern[i] = "<mask>"
    sampled = model.sample(
        all_coords, partial_seq=padding_pattern, temperature=temperature
    )
    sampled = sampled[:target_chain_len]
    return sampled


def score_sequence_in_complex(
    model, alphabet, coords, target_chain_id, target_seq, padding_length=10
):
    """
    Scores sequence for one chain in a complex.
    Args:
        model: An instance of the GVPTransformer model
        alphabet: Alphabet for the model
        coords: Dictionary mapping chain ids to L x 3 x 3 array for N, CA, C
            coordinates representing the backbone of each chain
        target_chain_id: The chain id to sample sequences for
        target_seq: Target sequence for the target chain for scoring.
        padding_length: padding length in between chains
    Returns:
        Tuple (ll_fullseq, ll_withcoord)
        - ll_fullseq: Average log-likelihood over the full target chain
        - ll_withcoord: Average log-likelihood in target chain excluding those
            residues without coordinates
    """
    all_coords = _concatenate_coords(coords, target_chain_id)

    loss, target_padding_mask = get_sequence_loss(
        model, alphabet, all_coords, target_seq
    )
    ll_fullseq = -np.sum(loss * ~target_padding_mask) / np.sum(~target_padding_mask)

    # Also calculate average when excluding masked portions
    coord_mask = np.all(np.isfinite(coords[target_chain_id]), axis=(-1, -2))
    ll_withcoord = -np.sum(loss * coord_mask) / np.sum(coord_mask)
    return ll_fullseq, ll_withcoord


def get_encoder_output_for_complex(model, alphabet, coords, target_chain_id):
    """
    Args:
        model: An instance of the GVPTransformer model
        alphabet: Alphabet for the model
        coords: Dictionary mapping chain ids to L x 3 x 3 array for N, CA, C
            coordinates representing the backbone of each chain
        target_chain_id: The chain id to sample sequences for
    Returns:
        Dictionary mapping chain id to encoder output for each chain
    """
    all_coords = _concatenate_coords(coords, target_chain_id)
    all_rep = get_encoder_output(model, alphabet, all_coords)
    target_chain_len = coords[target_chain_id].shape[0]
    return all_rep[:target_chain_len]


# MH
def get_ab_HL_coords_seq_pos_posins(pdb_path: str, Hchain: str, Lchain: str):
    """Get combined antibody H/L chain coords, seq, pos, posins from pdb_path"""

    coords_dict, seq_dict, pos_dict, posins_dict = load_complex_coords(
        pdb_path, [Hchain, Lchain]
    )
    (
        coords_concatenated,
        seq_concatenated,
        pos_concatenated,
        posins_concatenated,
    ) = concatenate_coords_HL(
        coords_dict, seq_dict, pos_dict, posins_dict, heavy_chain_id=Hchain
    )

    return coords_concatenated, seq_concatenated, pos_concatenated, posins_concatenated
