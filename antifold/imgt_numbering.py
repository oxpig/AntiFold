"""IMGT numbering: renumber antibody heavy/light chains with ANARCII."""

import logging
import os

import pandas as pd

from antifold.if1_dataset import get_pdb_path

log = logging.getLogger(__name__)

# AntiFold runs one heavy chain, optionally paired with one light chain. Only
# those two slots are renumbered - antigen and other chains are written as-is.
CHAIN_COLUMNS = ("Hchain", "Lchain")

# ANARCII reports kappa and lambda separately; AntiFold treats both as light
HEAVY_TYPE = "H"
LIGHT_TYPES = ("K", "L")

RENUMBERING_UNAVAILABLE = (
    "IMGT renumbering needs anarcii and gemmi ({error}): pip install anarcii gemmi\n"
    "Or re-run with --number_with_anarcii false to use the input numbering as-is"
)


class UnusableChains(ValueError):
    """This PDB's heavy/light chains cannot be used, so the PDB is skipped"""


def _candidate_chains(pdbs_csv, i):
    """The heavy/light chain slots for this row, as {column: chain ID}"""
    return {
        col: pdbs_csv.loc[i, col]
        for col in CHAIN_COLUMNS
        if col in pdbs_csv.columns and not pd.isna(pdbs_csv.loc[i, col])
    }


def _get_chain(structure, chain, pdb_path):
    found = structure[0].find_chain(chain)
    if found is None:
        raise UnusableChains(f"chain {chain} not found in {pdb_path}")
    return found


def _chain_type(pdb, chain, result, qa_passed):
    """ANARCII chain type, raising unless it is an antibody heavy or light chain"""

    if result["chain_type"] in (HEAVY_TYPE, *LIGHT_TYPES) and qa_passed:
        return result["chain_type"]

    raise UnusableChains(
        f"{pdb} chain {chain} is not an antibody heavy or light chain: ANARCII "
        f"reports chain type {result['chain_type']}, score {result['score']:.1f}"
        f"{', ' + result['error'] if result['error'] else ''}\n"
        f"Specify the antibody chains with --heavy_chain/--light_chain, or re-run "
        f"with --number_with_anarcii false to use the input numbering as-is"
    )


def _antibody_chain_types(pdb, numbered, qa):
    """ANARCII chain types of the antibody chains only; others are antigen/context"""

    chain_types = {
        chain: result["chain_type"]
        for chain, result in numbered.items()
        if result["chain_type"] in (HEAVY_TYPE, *LIGHT_TYPES) and qa(result)
    }
    if not chain_types:
        raise UnusableChains(
            f"no antibody chain found among {sorted(numbered)}. Specify the chains "
            f"with --heavy_chain/--light_chain, or re-run with "
            f"--number_with_anarcii false to use the input numbering as-is"
        )
    return chain_types


def _assign_chains(pdb, chain_types):
    """Maps Hchain/Lchain to chain IDs by ANARCII chain type"""

    assigned = {}
    for chain, chain_type in chain_types.items():
        col = "Hchain" if chain_type == HEAVY_TYPE else "Lchain"

        if col in assigned:
            name = "heavy" if col == "Hchain" else "light"
            raise UnusableChains(
                f"{pdb}: chains {assigned[col]} and {chain} are both {name} chains. "
                f"AntiFold runs one heavy chain, optionally paired with one light chain"
            )
        assigned[col] = chain

    if "Hchain" not in assigned:
        raise UnusableChains(
            f"{pdb}: no heavy chain among {sorted(chain_types)}. AntiFold requires "
            f"a heavy (or nanobody) chain"
        )

    return assigned


def _format_chains(chains):
    """Chain assignment as 'Hchain=H, Lchain=L' (chain IDs may be numpy strings)"""
    return ", ".join(f"{col}={chain}" for col, chain in sorted(chains.items()))


def _check_agreement(pdb, candidates, assigned):
    """Raises unless ANARCII agrees with the heavy/light chains the user gave"""

    if assigned == candidates:
        return

    raise UnusableChains(
        f"{pdb}: ANARCII disagrees with the chains given "
        f"({_format_chains(candidates)}): it reads them as "
        f"{_format_chains(assigned)}. Correct the chains, or re-run with "
        f"--number_with_anarcii false to use them as given"
    )


def renumber_pdbs(
    pdbs_csv,
    pdb_dir,
    out_dir,
    nanobody_mode=False,
    chains_specified=False,
    custom_chain_mode=False,
    device="cpu",
):
    """IMGT renumbers antibody heavy/light chains, returns (pdbs_csv, pdb_dir)

    ANARCII decides which chain is heavy and which is light. Chains the user
    named must match, or this raises; chains AntiFold guessed from file order
    are corrected. Antigen and other chains are written out untouched.
    """

    # Imported here so anarcii and gemmi stay optional
    try:
        import gemmi
        import torch
        from anarcii import Anarcii
        from anarcii.input_data_processing import polymer_seq
        from anarcii.pipeline import numbered_sequence_qa, renumber_pdbx
    except ImportError as e:
        raise ImportError(RENUMBERING_UNAVAILABLE.format(error=e)) from e

    os.makedirs(out_dir, exist_ok=True)

    # ncpu pins torch.set_num_threads(), which ANARCII otherwise raises to the core
    # count. Thread count reorders float summation in the encoder, so leaving it to
    # ANARCII makes AntiFold logits depend on whether renumbering ran.
    seq_type = "vhh" if nanobody_mode else "antibody"
    model = Anarcii(
        seq_type=seq_type,
        cpu=(device != "cuda"),
        ncpu=torch.get_num_threads(),
        verbose=False,
    )
    log.info(f"IMGT renumbering with ANARCII ({seq_type} model) to {out_dir}")

    pdbs_csv = pdbs_csv.copy()
    kept, skipped = [], []

    for i in pdbs_csv.index:
        _pdb = pdbs_csv.loc[i, "pdb"]
        pdb_path = get_pdb_path(pdb_dir, _pdb)

        structure = gemmi.read_structure(pdb_path)
        structure.setup_entities()

        # A PDB whose chains ANARCII cannot use is skipped, so one bad structure
        # does not lose a whole batch
        try:
            if chains_specified:
                # Only the chains given, and each one must be an antibody chain
                candidates = _candidate_chains(pdbs_csv, i)
                numbered = model.number(
                    {
                        chain: polymer_seq(_get_chain(structure, chain, pdb_path))
                        for chain in candidates.values()
                    }
                )
                chain_types = {
                    chain: _chain_type(_pdb, chain, result, numbered_sequence_qa(result))
                    for chain, result in numbered.items()
                }
            else:
                # No chains given: read every chain and let ANARCII find the antibody
                candidates = None
                polymers = {ch.name: polymer_seq(ch) for ch in structure[0]}
                numbered = model.number({c: s for c, s in polymers.items() if s})
                chain_types = _antibody_chain_types(_pdb, numbered, numbered_sequence_qa)

            assigned = _assign_chains(_pdb, chain_types)

            if chains_specified:
                _check_agreement(_pdb, candidates, assigned)
            else:
                if "Lchain" not in assigned and not custom_chain_mode:
                    raise UnusableChains(
                        f"only a heavy chain ({assigned['Hchain']}) found, and an "
                        f"unpaired chain needs --nanobody_chain or --custom_chain_mode"
                    )
                log.info(f"{_pdb}: ANARCII found {_format_chains(assigned)}")

        except UnusableChains as e:
            log.warning(f"WARNING: skipping {_pdb}: {e}")
            skipped.append(str(_pdb))
            continue

        if not chains_specified:
            for col in CHAIN_COLUMNS:
                if col in pdbs_csv.columns:
                    pdbs_csv.loc[i, col] = None

        for col, chain in assigned.items():
            pdbs_csv.loc[i, col] = chain
            renumber_pdbx(structure, 0, chain, numbered[chain])
            log.info(f"{_pdb} {col} {chain}: IMGT renumbered ({chain_types[chain]})")

        # AntiFold reads the first model only, so the rest would be written out
        # still carrying their original numbering
        while len(structure) > 1:
            del structure[1]

        out_path = f"{out_dir}/{os.path.basename(pdb_path)}"
        if out_path.endswith(".cif"):
            structure.make_mmcif_document().write_file(out_path)
        else:
            structure.write_pdb(out_path)

        kept.append(i)

    if skipped:
        log.warning(
            f"WARNING: skipped {len(skipped)}/{len(pdbs_csv)} PDBs on chain "
            f"mismatch: {', '.join(skipped)}"
        )

    if not kept:
        raise ValueError(
            f"No PDBs left to run: ANARCII could not use the chains of any of the "
            f"{len(pdbs_csv)} given. Re-run with --number_with_anarcii false to use "
            f"the chains and numbering as given"
        )

    # Reset index: InverseData indexes rows by position, via df.loc[i] over range(len(df))
    return pdbs_csv.loc[kept].reset_index(drop=True), out_dir
