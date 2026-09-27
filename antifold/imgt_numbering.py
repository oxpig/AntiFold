"""IMGT numbering: renumber antibody heavy/light chains with ANARCII."""

import logging
import os

import gemmi
import pandas as pd
from anarcii import Anarcii
from anarcii.input_data_processing import polymer_seq
from anarcii.pipeline import numbered_sequence_qa, renumber_pdbx

from antifold.if1_dataset import get_pdb_path

log = logging.getLogger(__name__)

# AntiFold runs one heavy chain, optionally paired with one light chain
CHAIN_COLUMNS = ("Hchain", "Lchain")

# ANARCII reports kappa and lambda separately; AntiFold treats both as light
HEAVY_TYPE = "H"
LIGHT_TYPES = ("K", "L")

# IMGT-numbered heavy chains have no position 10, so its presence means the input
# was never IMGT numbered and every region mask read off it would be wrong
NON_IMGT_HEAVY_POSITION = 10


class UnusableChains(ValueError):
    """This PDB's heavy/light chains cannot be used, so the PDB is skipped"""


def _candidate_chains(pdbs_csv, i):
    """The heavy/light chain slots for this row, as {column: chain ID}"""
    return {
        col: pdbs_csv.loc[i, col]
        for col in CHAIN_COLUMNS
        if col in pdbs_csv.columns and not pd.isna(pdbs_csv.loc[i, col])
    }


def _row_chains(pdbs_csv, i):
    """Chain IDs listed for this row, in the order InverseData reads them"""
    return [
        pdbs_csv.loc[i, col]
        for col in pdbs_csv.columns[1:]
        if "chain" in col and not pd.isna(pdbs_csv.loc[i, col])
    ]


def _get_chain(structure, chain, pdb_path):
    found = structure[0].find_chain(chain)
    if found is None:
        raise UnusableChains(f"chain {chain} not found in {pdb_path}")
    return found


def _number_chains(model, structure, chains, pdb_path):
    """ANARCII numbering of the named chains, keyed by chain ID"""

    sequences = {
        chain: polymer_seq(_get_chain(structure, chain, pdb_path)) for chain in chains
    }
    sequences = {chain: seq for chain, seq in sequences.items() if seq}

    if not sequences:
        raise UnusableChains(f"no polymer chains to number in {pdb_path}")

    return model.number(sequences)


def _chain_type(chain, result, qa_passed):
    """ANARCII chain type, raising unless it is an antibody heavy or light chain"""

    if result["chain_type"] in (HEAVY_TYPE, *LIGHT_TYPES) and qa_passed:
        return result["chain_type"]

    raise UnusableChains(
        f"chain {chain} is not an antibody heavy or light chain: ANARCII "
        f"reports chain type {result['chain_type']}, score {result['score']:.1f}"
        f"{', ' + result['error'] if result['error'] else ''}\n"
        f"Specify the antibody chains with --heavy_chain/--light_chain, or re-run "
        f"with --no_number_with_anarcii to use the input numbering as-is"
    )


def _antibody_chain_types(numbered, qa):
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
            f"--no_number_with_anarcii to use the input numbering as-is"
        )
    return chain_types


def _assign_chains(chain_types):
    """Maps Hchain/Lchain to chain IDs by ANARCII chain type"""

    assigned = {}
    for chain, chain_type in chain_types.items():
        col = "Hchain" if chain_type == HEAVY_TYPE else "Lchain"

        if col in assigned:
            name = "heavy" if col == "Hchain" else "light"
            raise UnusableChains(
                f"chains {assigned[col]} and {chain} are both {name} chains. "
                f"AntiFold runs one heavy chain, optionally paired with one light chain"
            )
        assigned[col] = chain

    if "Hchain" not in assigned:
        raise UnusableChains(
            f"no heavy chain among {sorted(chain_types)}. AntiFold requires "
            f"a heavy (or nanobody) chain"
        )

    return assigned


def _format_chains(chains):
    """Chain assignment as 'Hchain=H, Lchain=L' (chain IDs may be numpy strings)"""
    return ", ".join(f"{col}={chain}" for col, chain in sorted(chains.items()))


def _check_agreement(candidates, assigned):
    """Raises unless ANARCII agrees with the heavy/light chains the user gave"""

    if assigned == candidates:
        return

    raise UnusableChains(
        f"ANARCII disagrees with the chains given "
        f"({_format_chains(candidates)}): it reads them as "
        f"{_format_chains(assigned)}. Correct the chains, or re-run with "
        f"--no_number_with_anarcii to use them as given"
    )


def renumber_pdbs(
    pdbs_csv,
    pdb_dir,
    out_dir,
    nanobody_mode=False,
    chains_specified=False,
    custom_chain_mode=False,
    device="cpu",
    num_threads=1,
):
    """IMGT renumbers antibody heavy/light chains, returns (pdbs_csv, pdb_dir)

    Every antibody chain ANARCII recognises is renumbered; antigen and other
    chains are written out untouched. Which chain counts as heavy or light is
    decided by how the chains reached us: named Hchain/Lchain must be the type
    claimed or this raises, chain columns under other names in custom_chain_mode
    are a selection and are left alone, and chains AntiFold guessed from file
    order are corrected to ANARCII's reading.
    """

    os.makedirs(out_dir, exist_ok=True)

    # ANARCII raises torch's thread count to the core count unless ncpu is given, and
    # the thread count reorders float summation in the encoder - so AntiFold logits
    # would depend on whether renumbering ran.
    seq_type = "vhh" if nanobody_mode else "antibody"
    model = Anarcii(
        seq_type=seq_type,
        cpu=(device != "cuda"),
        ncpu=num_threads,
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
            candidates = _candidate_chains(pdbs_csv, i) if chains_specified else {}

            if candidates:
                # Named Hchain/Lchain are a claim: each must be the type claimed
                numbered = _number_chains(
                    model, structure, candidates.values(), pdb_path
                )
                chain_types = {
                    chain: _chain_type(chain, result, numbered_sequence_qa(result))
                    for chain, result in numbered.items()
                }
                _check_agreement(candidates, _assign_chains(chain_types))
                reassign = None

            elif custom_chain_mode:
                # Chains listed under other column names are a selection, not a
                # heavy/light claim: renumber only the antibody chains among them
                numbered = _number_chains(
                    model, structure, _row_chains(pdbs_csv, i), pdb_path
                )
                chain_types = _antibody_chain_types(numbered, numbered_sequence_qa)
                reassign = None

            else:
                # No chains given: read every chain and let ANARCII find the antibody
                polymers = {ch.name: polymer_seq(ch) for ch in structure[0]}
                numbered = model.number({c: s for c, s in polymers.items() if s})
                chain_types = _antibody_chain_types(numbered, numbered_sequence_qa)
                reassign = _assign_chains(chain_types)

                if "Lchain" not in reassign:
                    raise UnusableChains(
                        f"only a heavy chain ({reassign['Hchain']}) found, and an "
                        f"unpaired chain needs --nanobody_chain or --custom_chain_mode"
                    )
                log.info(f"{_pdb}: ANARCII found {_format_chains(reassign)}")

        except UnusableChains as e:
            log.warning(f"WARNING: skipping {_pdb}: {e}")
            skipped.append(str(_pdb))
            continue

        if reassign:
            for col in CHAIN_COLUMNS:
                if col in pdbs_csv.columns:
                    pdbs_csv.loc[i, col] = None
            for col, chain in reassign.items():
                pdbs_csv.loc[i, col] = chain

        for chain, chain_type in chain_types.items():
            renumber_pdbx(structure, 0, chain, numbered[chain])
            log.info(f"{_pdb} chain {chain}: IMGT renumbered ({chain_type})")

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
            f"{len(pdbs_csv)} given. Re-run with --no_number_with_anarcii to use "
            f"the chains and numbering as given"
        )

    # Reset index: InverseData indexes rows by position, via df.loc[i] over range(len(df))
    return pdbs_csv.loc[kept].reset_index(drop=True), out_dir


def warn_if_not_imgt_numbered(pdbs_csv, pdb_dir):
    """Warns for each PDB whose heavy chain is not IMGT numbered

    Renumbering is off, so region masks are read off the input numbering as given.
    """

    for i in pdbs_csv.index:
        _pdb = pdbs_csv.loc[i, "pdb"]
        chains = _row_chains(pdbs_csv, i)
        if not chains:
            raise ValueError(f"No chains listed for PDB {_pdb}")

        pdb_path = get_pdb_path(pdb_dir, _pdb)
        structure = gemmi.read_structure(pdb_path)
        heavy = _get_chain(structure, str(chains[0]), pdb_path)

        if any(res.seqid.num == NON_IMGT_HEAVY_POSITION for res in heavy):
            log.warning(
                f"WARNING: {_pdb} chain {chains[0]} has IMGT position "
                f"{NON_IMGT_HEAVY_POSITION}, which IMGT-numbered heavy chains do not, "
                f"so it is probably not IMGT numbered. Region masks and sampling will "
                f"be wrong. Drop --no_number_with_anarcii to renumber it with ANARCII"
            )
