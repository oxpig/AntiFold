# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

log = logging.getLogger(__name__)

from typing import List, Sequence, Tuple

import biotite.structure
import numpy as np
import torch
import torch.nn.functional as F
from biotite.sequence import ProteinSequence
from biotite.structure import filter_backbone, get_chains
from biotite.structure.io import pdb, pdbx
from biotite.structure.residues import get_residues


def load_structure_fast(fpath, chain=None):
    """
    Args:
        fpath: filepath to either pdb or cif file
        chain: the chain id or list of chain ids to load
    Returns:
        [biotite.structure.AtomArray]
    """

    if fpath[-3:] == "pdb":
        with open(fpath) as fin:
            pdbf = pdb.PDBFile.read(fin)
            structure_full = pdbf.get_structure(model=1, extra_fields=["b_factor"])
    else:
        raise Exception

    bbmask = filter_backbone(structure_full)
    structure = structure_full[bbmask]

    all_chains = get_chains(structure)

    if len(all_chains) == 0:
        raise ValueError("No chains found in the input file.")

    if chain is None:
        chain_ids = all_chains

    elif isinstance(chain, list):
        chain_ids = chain

    else:
        chain_ids = [chain]

    for chain in chain_ids:
        if chain not in all_chains:
            raise ValueError(f"Chain {chain} not found in input file")

    chain_filter = [a.chain_id in chain_ids for a in structure]
    structure = structure[chain_filter]

    chain_filter_full = [a.chain_id in chain_ids for a in structure_full]
    structure_full = structure_full[chain_filter_full]

    return structure, structure_full


def load_structure(fpath, chain=None):
    """
    Args:
        fpath: filepath to either pdb or cif file
        chain: the chain id or list of chain ids to load
    Returns:
        biotite.structure.AtomArray
    """
    if fpath.endswith("cif"):
        with open(fpath) as fin:
            pdbxf = pdbx.PDBxFile.read(fin)
        structure = pdbx.get_structure(pdbxf, model=1)
    elif fpath.endswith("pdb"):
        with open(fpath) as fin:
            pdbf = pdb.PDBFile.read(fin)
        structure = pdb.get_structure(pdbf, model=1)
    bbmask = filter_backbone(structure)
    structure = structure[bbmask]
    all_chains = get_chains(structure)
    if len(all_chains) == 0:
        raise ValueError("No chains found in the input file.")
    if chain is None:
        chain_ids = all_chains
    elif isinstance(chain, list):
        chain_ids = chain
    else:
        chain_ids = [chain]
    for chain in chain_ids:
        if chain not in all_chains:
            raise ValueError(f"Chain {chain} not found in input file")
    chain_filter = [a.chain_id in chain_ids for a in structure]
    structure = structure[chain_filter]
    return structure


def extract_coords_from_structure(structure: biotite.structure.AtomArray):
    """
    Args:
        structure: An instance of biotite AtomArray
    Returns:
        Tuple (coords, seq)
            - coords is an L x 3 x 3 array for N, CA, C coordinates
            - seq is the extracted sequence
    """
    coords = get_atom_coords_residuewise(["N", "CA", "C"], structure)
    residue_identities = get_residues(structure)[1]
    seq = "".join([ProteinSequence.convert_letter_3to1(r) for r in residue_identities])
    return coords, seq


def extract_coords_seq_pos_from_structure(structure: biotite.structure.AtomArray):
    """
    Args:
        structure: An instance of biotite AtomArray
    Returns:
        Tuple (coords, seq, res_pos)
            - coords is an L x 3 x 3 array for N, CA, C coordinates
            - seq is the extracted sequence
            - res_pos is a L array for residue positions
              (duplicates for 111, 111A, 111B etc)
    """

    coords = get_atom_coords_residuewise(["N", "CA", "C"], structure)
    res_pos, res_3letter = get_residues(structure)
    seq = "".join([ProteinSequence.convert_letter_3to1(r) for r in res_3letter])
    return coords, seq, res_pos


def load_coords(fpath, chain):
    """
    Args:
        fpath: filepath to either pdb or cif file
        chain: the chain id
    Returns:
        Tuple (coords, seq)
            - coords is an L x 3 x 3 array for N, CA, C coordinates
            - seq is the extracted sequence
    """
    structure = load_structure(fpath, chain)
    return extract_coords_from_structure(structure)


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


def rotate(v, R):
    """
    Rotates a vector by a rotation matrix.

    Args:
        v: 3D vector, tensor of shape (length x batch_size x channels x 3)
        R: rotation matrix, tensor of shape (length x batch_size x 3 x 3)

    Returns:
        Rotated version of v by rotation matrix R.
    """
    R = R.unsqueeze(-3)
    v = v.unsqueeze(-1)
    return torch.sum(v * R, dim=-2)


def get_rotation_frames(coords):
    """
    Returns a local rotation frame defined by N, CA, C positions.

    Args:
        coords: coordinates, tensor of shape (batch_size x length x 3 x 3)
        where the third dimension is in order of N, CA, C

    Returns:
        Local relative rotation frames in shape (batch_size x length x 3 x 3)
    """
    v1 = coords[:, :, 2] - coords[:, :, 1]
    v2 = coords[:, :, 0] - coords[:, :, 1]
    e1 = normalize(v1, dim=-1)
    u2 = v2 - e1 * torch.sum(e1 * v2, dim=-1, keepdim=True)
    e2 = normalize(u2, dim=-1)
    e3 = torch.cross(e1, e2, dim=-1)
    R = torch.stack([e1, e2, e3], dim=-2)
    return R


def nan_to_num(ts, val=0.0):
    """
    Replaces nans in tensor with a fixed value.
    """
    val = torch.tensor(val, dtype=ts.dtype, device=ts.device)
    return torch.where(~torch.isfinite(ts), val, ts)


def rbf(values, v_min, v_max, n_bins=16):
    """
    Returns RBF encodings in a new dimension at the end.
    """
    rbf_centers = torch.linspace(v_min, v_max, n_bins, device=values.device)
    rbf_centers = rbf_centers.view([1] * len(values.shape) + [-1])
    rbf_std = (v_max - v_min) / n_bins
    # v_expand = torch.unsqueeze(values, -1)
    z = (values.unsqueeze(-1) - rbf_centers) / rbf_std
    return torch.exp(-(z ** 2))


def norm(tensor, dim, eps=1e-8, keepdim=False):
    """
    Returns L2 norm along a dimension.
    """
    return torch.sqrt(torch.sum(torch.square(tensor), dim=dim, keepdim=keepdim) + eps)


def normalize(tensor, dim=-1):
    """
    Normalizes a tensor along a dimension after removing nans.
    """
    return nan_to_num(torch.div(tensor, norm(tensor, dim=dim, keepdim=True)))


class BatchConverter(object):
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """

    def __init__(self, alphabet, truncation_seq_length: int = None):
        self.alphabet = alphabet
        self.truncation_seq_length = truncation_seq_length

    def __call__(self, raw_batch: Sequence[Tuple[str, str]]):
        # RoBERTa uses an eos token, while ESM-1 does not.
        batch_size = len(raw_batch)
        batch_labels, seq_str_list = zip(*raw_batch)
        seq_encoded_list = [self.alphabet.encode(seq_str) for seq_str in seq_str_list]
        if self.truncation_seq_length:
            seq_encoded_list = [
                seq_str[: self.truncation_seq_length] for seq_str in seq_encoded_list
            ]
        max_len = max(len(seq_encoded) for seq_encoded in seq_encoded_list)
        tokens = torch.empty(
            (
                batch_size,
                max_len
                + int(self.alphabet.prepend_bos)
                + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens.fill_(self.alphabet.padding_idx)
        labels = []
        strs = []

        for i, (label, seq_str, seq_encoded) in enumerate(
            zip(batch_labels, seq_str_list, seq_encoded_list)
        ):
            labels.append(label)
            strs.append(seq_str)
            if self.alphabet.prepend_bos:
                tokens[i, 0] = self.alphabet.cls_idx
            seq = torch.tensor(seq_encoded, dtype=torch.int64)
            tokens[
                i,
                int(self.alphabet.prepend_bos) : len(seq_encoded)
                + int(self.alphabet.prepend_bos),
            ] = seq
            if self.alphabet.append_eos:
                tokens[
                    i, len(seq_encoded) + int(self.alphabet.prepend_bos)
                ] = self.alphabet.eos_idx

        return labels, strs, tokens


class CoordBatchConverter_mask_gpu(BatchConverter):
    def __call__(self, raw_batch: Sequence[Tuple[Sequence, str]], device=None):
        """
        Args:
            raw_batch: List of tuples (coords, confidence, seq)
            In each tuple,
                coords: list of floats, shape L x 3 x 3
                confidence: list of floats, shape L; or scalar float; or None
                seq: string of length L
        Returns:
            coords: Tensor of shape batch_size x L x 3 x 3
            confidence: Tensor of shape batch_size x L
            strs: list of strings
            tokens: LongTensor of shape batch_size x L
            padding_mask: ByteTensor of shape batch_size x L
        """
        self.alphabet.cls_idx = self.alphabet.get_idx("<cath>")
        batch = []
        loss_masks = []
        pos_list = []
        posinschain_list = []
        targets_list = []
        for (
            coords,
            confidence,
            seq,
            res_pos,
            res_posinschain,
            loss_mask,
            targets,
        ) in raw_batch:
            if confidence is None:
                confidence = 1.0
            if isinstance(confidence, float) or isinstance(confidence, int):
                confidence = [float(confidence)] * len(coords)
            if seq is None:
                seq = "X" * len(coords)
            batch.append(((coords, confidence), seq))
            loss_masks.append(loss_mask)
            pos_list.append(res_pos)
            posinschain_list.append(res_posinschain)
            targets_list.append(targets)

        coords_and_confidence, strs, tokens = super().__call__(batch)

        # pad beginning and end of each protein due to legacy reasons
        coords = [
            F.pad(torch.tensor(cd), (0, 0, 0, 0, 1, 1), value=np.inf)
            for cd, _ in coords_and_confidence
        ]
        confidence = [
            F.pad(torch.tensor(cf), (1, 1), value=-1.0)
            for _, cf in coords_and_confidence
        ]
        loss_masks = [F.pad(torch.tensor(cf), (1, 1), value=0) for cf in loss_masks]

        coords = self.collate_dense_tensors(coords, pad_v=np.nan)
        confidence = self.collate_dense_tensors(confidence, pad_v=-1.0)
        loss_masks = self.collate_dense_tensors(loss_masks, pad_v=np.nan)

        # Custom stack #MH
        res_pos = self.batch_convert_vals(pos_list)
        targets = self.batch_convert_vals(targets_list)

        # check for cuda
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if device == torch.device("cuda"):  # MH
            coords = coords.type(torch.float32).to(device)
            confidence = confidence.to(device)
            tokens = tokens.to(device)
            loss_masks = loss_masks.to(device)
            res_pos = res_pos.to(device)
            targets = targets.to(device)

        padding_mask = torch.isnan(coords[:, :, 0, 0])
        coord_mask = torch.isfinite(coords.sum(-2).sum(-1))
        confidence = confidence * coord_mask + (-1.0) * padding_mask

        # Also padding mask #MH
        if device == torch.device("cuda"):  # MH
            padding_mask = padding_mask.to(device)
            coord_mask = coord_mask.to(device)
            confidence = confidence.to(device)

        return (
            coords,
            confidence,
            strs,
            tokens,
            padding_mask,
            loss_masks,
            res_pos,
            posinschain_list,
            targets,
        )

    def batch_convert_vals(self, val_list):
        """
        Converts list of np.arrays to stacked torch tensor ESM-IF1 style
        List of np.arrays -> torch.tensor size (batch_size, max_len + 2), where +2 is for bos, eos
        """

        batch_size = len(val_list)
        max_len = max(len(v) for v in val_list)
        out = torch.full(size=(batch_size, max_len + 2), fill_value=np.nan,)  # bos, eos
        for i, rp in enumerate(val_list):
            out[i, 1 : len(rp) + 1] = torch.tensor(np.array(rp))

        return out

    def from_lists(self, coords_list, confidence_list=None, seq_list=None, device=None):
        """
        Args:
            coords_list: list of length batch_size, each item is a list of
            floats in shape L x 3 x 3 to describe a backbone
            confidence_list: one of
                - None, default to highest confidence
                - list of length batch_size, each item is a scalar
                - list of length batch_size, each item is a list of floats of
                    length L to describe the confidence scores for the backbone
                    with values between 0. and 1.
            seq_list: either None or a list of strings
        Returns:
            coords: Tensor of shape batch_size x L x 3 x 3
            confidence: Tensor of shape batch_size x L
            strs: list of strings
            tokens: LongTensor of shape batch_size x L
            padding_mask: ByteTensor of shape batch_size x L
        """
        batch_size = len(coords_list)
        if confidence_list is None:
            confidence_list = [None] * batch_size
        if seq_list is None:
            seq_list = [None] * batch_size
        raw_batch = zip(coords_list, confidence_list, seq_list)
        return self.__call__(raw_batch, device)

    @staticmethod
    def collate_dense_tensors(samples, pad_v):
        """
        Takes a list of tensors with the following dimensions:
            [(d_11,       ...,           d_1K),
             (d_21,       ...,           d_2K),
             ...,
             (d_N1,       ...,           d_NK)]
        and stack + pads them into a single tensor of:
        (N, max_i=1,N { d_i1 }, ..., max_i=1,N {diK})
        """
        if len(samples) == 0:
            return torch.Tensor()
        if len(set(x.dim() for x in samples)) != 1:
            raise RuntimeError(
                f"Samples has varying dimensions: {[x.dim() for x in samples]}"
            )
        (device,) = tuple(set(x.device for x in samples))  # assumes all on same device
        max_shape = [max(lst) for lst in zip(*[x.shape for x in samples])]
        result = torch.empty(
            len(samples), *max_shape, dtype=samples[0].dtype, device=device
        )
        result.fill_(pad_v)
        for i in range(len(samples)):
            result_i = result[i]
            t = samples[i]
            result_i[tuple(slice(0, k) for k in t.shape)] = t
        return result


class CoordBatchConverter_mask(BatchConverter):
    def __call__(self, raw_batch: Sequence[Tuple[Sequence, str]], device=None):
        """
        Args:
            raw_batch: List of tuples (coords, confidence, seq)
            In each tuple,
                coords: list of floats, shape L x 3 x 3
                confidence: list of floats, shape L; or scalar float; or None
                seq: string of length L
        Returns:
            coords: Tensor of shape batch_size x L x 3 x 3
            confidence: Tensor of shape batch_size x L
            strs: list of strings
            tokens: LongTensor of shape batch_size x L
            padding_mask: ByteTensor of shape batch_size x L
        """
        self.alphabet.cls_idx = self.alphabet.get_idx("<cath>")
        batch = []
        loss_masks = []
        pos_list = []
        posinschain_list = []
        targets_list = []
        for (
            coords,
            confidence,
            seq,
            res_pos,
            res_posinschain,
            loss_mask,
            targets,
        ) in raw_batch:
            if confidence is None:
                confidence = 1.0
            if isinstance(confidence, float) or isinstance(confidence, int):
                confidence = [float(confidence)] * len(coords)
            if seq is None:
                seq = "X" * len(coords)
            batch.append(((coords, confidence), seq))
            loss_masks.append(loss_mask)
            pos_list.append(res_pos)
            posinschain_list.append(res_posinschain)
            targets_list.append(targets)

        coords_and_confidence, strs, tokens = super().__call__(batch)

        # pad beginning and end of each protein due to legacy reasons
        coords = [
            F.pad(torch.tensor(cd), (0, 0, 0, 0, 1, 1), value=np.inf)
            for cd, _ in coords_and_confidence
        ]
        confidence = [
            F.pad(torch.tensor(cf), (1, 1), value=-1.0)
            for _, cf in coords_and_confidence
        ]
        loss_masks = [F.pad(torch.tensor(cf), (1, 1), value=0) for cf in loss_masks]

        coords = self.collate_dense_tensors(coords, pad_v=np.nan)
        confidence = self.collate_dense_tensors(confidence, pad_v=-1.0)
        loss_masks = self.collate_dense_tensors(loss_masks, pad_v=np.nan)

        # Custom stack #MH
        res_pos = self.batch_convert_vals(pos_list)
        targets = self.batch_convert_vals(targets_list)

        # check for cuda
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        coords = coords.type(torch.float32)
        padding_mask = torch.isnan(coords[:, :, 0, 0])
        coord_mask = torch.isfinite(coords.sum(-2).sum(-1))
        confidence = confidence * coord_mask + (-1.0) * padding_mask

        return (
            coords,
            confidence,
            strs,
            tokens,
            padding_mask,
            loss_masks,
            res_pos,
            posinschain_list,
            targets,
        )

    def batch_convert_vals(self, val_list):
        """
        Converts list of np.arrays to stacked torch tensor ESM-IF1 style
        List of np.arrays -> torch.tensor size (batch_size, max_len + 2), where +2 is for bos, eos
        """

        batch_size = len(val_list)
        max_len = max(len(v) for v in val_list)
        out = torch.full(size=(batch_size, max_len + 2), fill_value=np.nan,)  # bos, eos
        for i, rp in enumerate(val_list):
            out[i, 1 : len(rp) + 1] = torch.tensor(np.array(rp))

        return out

    def from_lists(self, coords_list, confidence_list=None, seq_list=None, device=None):
        """
        Args:
            coords_list: list of length batch_size, each item is a list of
            floats in shape L x 3 x 3 to describe a backbone
            confidence_list: one of
                - None, default to highest confidence
                - list of length batch_size, each item is a scalar
                - list of length batch_size, each item is a list of floats of
                    length L to describe the confidence scores for the backbone
                    with values between 0. and 1.
            seq_list: either None or a list of strings
        Returns:
            coords: Tensor of shape batch_size x L x 3 x 3
            confidence: Tensor of shape batch_size x L
            strs: list of strings
            tokens: LongTensor of shape batch_size x L
            padding_mask: ByteTensor of shape batch_size x L
        """
        batch_size = len(coords_list)
        if confidence_list is None:
            confidence_list = [None] * batch_size
        if seq_list is None:
            seq_list = [None] * batch_size
        raw_batch = zip(coords_list, confidence_list, seq_list)
        return self.__call__(raw_batch, device)

    @staticmethod
    def collate_dense_tensors(samples, pad_v):
        """
        Takes a list of tensors with the following dimensions:
            [(d_11,       ...,           d_1K),
             (d_21,       ...,           d_2K),
             ...,
             (d_N1,       ...,           d_NK)]
        and stack + pads them into a single tensor of:
        (N, max_i=1,N { d_i1 }, ..., max_i=1,N {diK})
        """
        if len(samples) == 0:
            return torch.Tensor()
        if len(set(x.dim() for x in samples)) != 1:
            raise RuntimeError(
                f"Samples has varying dimensions: {[x.dim() for x in samples]}"
            )
        (device,) = tuple(set(x.device for x in samples))  # assumes all on same device
        max_shape = [max(lst) for lst in zip(*[x.shape for x in samples])]
        result = torch.empty(
            len(samples), *max_shape, dtype=samples[0].dtype, device=device
        )
        result.fill_(pad_v)
        for i in range(len(samples)):
            result_i = result[i]
            t = samples[i]
            result_i[tuple(slice(0, k) for k in t.shape)] = t
        return result
