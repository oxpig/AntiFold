import logging

log = logging.getLogger(__name__)

import sys

sys.path.insert(0, ".")

import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch

from antifold.esm_multichain_util_custom import (concatenate_coords_any,
                                                 concatenate_coords_HL,
                                                 load_complex_coords)


class InverseData(torch.utils.data.Dataset):
    """
    Prepare dataset for ESM-IF1, including span masking and adding gaussian noise, returning
        batches with coords, confidence, strs, tokens, padding_mask
    Returns: coords, confidence, seq, res_pos, loss_mask, targets
    """

    def __init__(
        self,
        verbose: int = 0,
        loss_mask_flag: bool = False,
        gaussian_noise_flag: bool = True,
        gaussian_scale_A: float = 0.1,
        custom_chain_mode: bool = False,
    ):
        # Variables
        self.loss_mask_flag = loss_mask_flag
        self.gaussian_noise_flag = gaussian_noise_flag
        self.gaussian_scale_A = gaussian_scale_A
        self.seq_pdb_fasta_mismatches = 0
        self.custom_chain_mode = custom_chain_mode
        self.verbose = verbose

        if self.custom_chain_mode:
            log.info(
                "NOTE: Custom chain mode enabled. Will run specified chain(s)."
            )

    def load_coords_HL(
        self, pdb_path: str, Hchain: str, Lchain: str
    ) -> Tuple[np.array, str, np.array]:
        """Read pdb file and extract coordinates of backbone (N, CA, C) atoms of given chains
        Args:
            file path: path to pdb file
            Hchain: heavy chain
            Lchain: light chain
        Returns:
            coords_concatenated: 3d backbone coordinates of extracted structure w/ padding - padded coords set to inf
            seq_concatenated: AA sequence w/ padding
            pos_concatenated: residue positions w/ padding
        """

        # coords, seq = esm.inverse_folding.util.load_coords(fpath=pdb_path, chain=chain)
        coords_dict, seq_dict, pos_dict, posinschain_dict = load_complex_coords(
            pdb_path, [Hchain, Lchain]
        )
        (
            coords_concatenated,
            seq_concatenated,
            pos_concatenated,
            posinschain_concatenated,
        ) = concatenate_coords_HL(
            coords_dict, seq_dict, pos_dict, posinschain_dict, heavy_chain_id=Hchain
        )

        # Limit to IMGT VH/VL regions (pos 1-128)

        return (
            coords_concatenated,
            seq_concatenated,
            pos_concatenated,
            posinschain_concatenated,
        )

    def load_coords_any(
        self, pdb_path: str, chains: list,
    ) -> Tuple[np.array, str, np.array]:
        """Read pdb file and extract coordinates of backbone (N, CA, C) atoms of given chains
        Args:
            file path: path to pdb file
            chains: List of chains ids to extract
        Returns:
            coords_concatenated: 3d backbone coordinates of extracted structure w/ padding - padded coords set to inf
            seq_concatenated: AA sequence w/ padding
            pos_concatenated: residue positions w/ padding
        """

        # Get all chains
        coords_dict, seq_dict, pos_dict, posinschain_dict = load_complex_coords(
            pdb_path, chains,
        )

        # Concatenate
        (
            coords_concatenated,
            seq_concatenated,
            pos_concatenated,
            posinschain_concatenated,
        ) = concatenate_coords_any(
            coords_dict, seq_dict, pos_dict, posinschain_dict, chains,
        )

        # Limit to IMGT VH/VL regions (pos 1-128)

        return (
            coords_concatenated,
            seq_concatenated,
            pos_concatenated,
            posinschain_concatenated,
        )

    def add_gaussian_noise(self, coords: np.array, scale=0.1):
        """Add Gaussian noise at scale 0.1A to each coordinate
        Args:
            coords: 3d backbone coordinates of full_structure (as formatted by load_coords, can be span-masked)
        Returns:
            coords with Gaussian noise added
        """
        return coords + np.random.normal(scale=scale, size=coords.shape)

    def populate(
        self,
        pdbs_csv_or_dataframe: 'str, "path or pd.DataFrame"',
        pdb_dir: str,
        verbose: int = 1,
    ):
        """
        Gets the actual PDB paths to be used for training and testing,
        will filter on the PDBs present in the paragraph CSV dict if set.

        Args:
            pdbs_csv_or_dataframe: path to csv file containing pdb, Hchain and Lchain
        """

        # Accept DataFrame or CSV path
        if isinstance(pdbs_csv_or_dataframe, pd.DataFrame):
            df = pdbs_csv_or_dataframe
            log.info(f"Reading in ({len(df)}) PDBs from DataFrame")
        else:
            if not os.path.exists(pdbs_csv_or_dataframe):
                log.error(
                    f"Unable to find pdbs_csv_or_dataframe {pdbs_csv_or_dataframe}"
                )
                sys.exit(1)

            df = pd.read_csv(pdbs_csv_or_dataframe)
            log.info(f"Populating {len(df)} PDBs from {pdbs_csv_or_dataframe}")

        if not len(df) >= 1:
            log.error(f"CSV file {pdbs_csv_or_dataframe} must contain at least 1 PDB")
            sys.exit(1)

        if (
            not df.columns.isin(["pdb", "Hchain", "Lchain"]).sum() >= 3
            and not self.custom_chain_mode
        ):
            log.error(
                f"CSV file requires columns 'pdb, Hchain, Lchain': found {df.columns}"
            )
            sys.exit(1)

        # Create list of PDB paths and check that they exist
        pdb_path_list = []
        for _pdb in df["pdb"]:
            pdb_path = f"{pdb_dir}/{_pdb}.pdb"

            # Check for PDB/CIF
            pdb_path = pdb_path if os.path.exists(pdb_path) else f"{pdb_dir}/{_pdb}.cif"
            pdb_path_list.append(pdb_path)

            if not os.path.exists(pdb_path):
                raise Exception(f"Unable to find PDB/CIF file: {pdb_path}")

        # Infer order of chain from CSV columns (first item pdb, then chains)
        # Should be Hchain, Lchain, then any order
        chain_order = [c for c in df.columns[1:] if "chain" in c]

        # Construct info dict
        self.pdb_info_dict = {}

        for i in range(len(df)):
            _pdb = df.loc[i, "pdb"]
            _pdb_path = pdb_path_list[i]

            # Extract chains present in CSV
            chains = df.loc[i, chain_order]
            chains = [c for c in chains if not pd.isna(c)]
            if len(chains) == 0:
                raise Exception(f"No chains found for PDB {_pdb}")

            # Infer heavy and light chain
            if len(chains) >= 2:
                Hchain, Lchain = chains[0], chains[1]
            # Infer nanobody
            else:
                Hchain = chains[0]
                Lchain = None

            # PDB name with chains
            _pdb_chainsname = _pdb + "_" + "".join(chains)

            self.pdb_info_dict[i] = {
                "pdb": _pdb,
                "pdb_path": _pdb_path,
                "chain_order": chains,
                "pdb_chainsname": _pdb_chainsname,
                "Hchain": Hchain,
                "Lchain": Lchain,
            }

    def __getitem__(self, idx: int):
        """
        # obtain pdb_info for entry with index idx - pdb_info contains the pdb_path (generated in populate)
        Format data to pass to PyTorch DataLoader (with collate_fn = util.CoordBatchConverter)
        """

        # Experimental - use any chains (e.g. with antigen or for ESM-IF1)
        if self.custom_chain_mode:
            pdb_path = self.pdb_info_dict[idx]["pdb_path"]
            chains = self.pdb_info_dict[idx]["chain_order"]

            coords, seq_pdb, pos_pdb, pos_pdb_arr_str = self.load_coords_any(
                pdb_path, chains
            )

        # Regular mode
        else:
            pdb_path = self.pdb_info_dict[idx]["pdb_path"]
            Hchain = self.pdb_info_dict[idx]["Hchain"]
            Lchain = self.pdb_info_dict[idx]["Lchain"]

            coords, seq_pdb, pos_pdb, pos_pdb_arr_str = self.load_coords_HL(
                pdb_path, Hchain, Lchain,
            )

        # Not used, included for legacy reasons
        targets = np.full(len(pos_pdb_arr_str), np.nan)

        # Add (0.1 Å) gaussian noise to Ca, C, N co-ordinates
        if self.gaussian_noise_flag:
            coords = self.add_gaussian_noise(coords=coords, scale=self.gaussian_scale_A)

        # Initialize empty loss_mask
        loss_mask = np.full(len(coords), fill_value=False)

        # reset padding coords between H and L to np.nan (instead of np.inf, which was set so as not to interfere with masking)
        coords[pos_pdb_arr_str == "nan"] = np.nan

        # confidence currently not taken - update
        confidence = None

        # check masking done correctly - no masking in linker region (between heavy and light chains)
        assert loss_mask[np.where(pos_pdb_arr_str == "nan")].sum() == 0

        return coords, confidence, seq_pdb, pos_pdb, pos_pdb_arr_str, loss_mask, targets

    def __len__(self):
        """Get number of entries in dataset"""
        return len(self.pdb_info_dict)
