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
CHAIN_NAMES = {"Hchain": "a heavy chain", "Lchain": "a light chain"}

# ANARCII reports kappa and lambda separately; AntiFold treats both as light
HEAVY_TYPE = "H"
LIGHT_TYPES = ("K", "L")

# IMGT-numbered heavy chains have no position 10, so its presence means the input
# was never IMGT numbered and every region mask read off it would be wrong
NON_IMGT_HEAVY_POSITION = 10

AS_GIVEN = (
    "Correct the chains, or re-run with --skip_anarcii_numbering to use them as given"
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


def _number_chains(model, structure, pdb_path):
    """ANARCII numbering of every polymer chain in the structure, by chain ID"""

    sequences = {chain.name: polymer_seq(chain) for chain in structure[0]}
    sequences = {chain: seq for chain, seq in sequences.items() if seq}

    if not sequences:
        raise UnusableChains(f"no polymer chains to number in {pdb_path}")

    return model.number(sequences)


def _antibody_chains(numbered):
    """{chain ID: ANARCII type} for the chains read as antibody heavy/light chains

    Everything else - antigens, and anything failing ANARCII's own QA - is left out,
    so it keeps the numbering it came with.
    """
    return {
        chain: result["chain_type"]
        for chain, result in numbered.items()
        if result["chain_type"] in (HEAVY_TYPE, *LIGHT_TYPES)
        and numbered_sequence_qa(result)
    }


def _assign_chains(antibody):
    """Maps Hchain/Lchain to chain IDs by ANARCII chain type"""

    assigned = {}
    for chain, chain_type in antibody.items():
        col = "Hchain" if chain_type == HEAVY_TYPE else "Lchain"

        if col in assigned:
            raise UnusableChains(
                f"chains {assigned[col]} and {chain} are both {CHAIN_NAMES[col]}. "
                f"AntiFold runs one heavy chain, optionally paired with one light chain"
            )
        assigned[col] = chain

    if "Hchain" not in assigned:
        raise UnusableChains(
            f"no antibody heavy chain found (ANARCII recognised "
            f"{sorted(antibody) or 'no antibody chains'}). AntiFold requires a heavy "
            f"(or nanobody) chain"
        )

    return assigned


def _format_chains(chains):
    """Chain assignment as 'Hchain=H, Lchain=L' (chain IDs may be numpy strings)"""
    return ", ".join(f"{col}={chain}" for col, chain in sorted(chains.items()))


def _check_named_chains(named, numbered, antibody, custom_chain_mode):
    """Raises unless each named Hchain/Lchain is the antibody chain type claimed

    In custom_chain_mode the chain columns are a selection that may hold any chain,
    so one ANARCII does not recognise keeps its own numbering. A chain read as the
    other type is a contradiction either way.
    """

    for col, chain in named.items():
        if chain not in numbered:
            raise UnusableChains(f"{col}={chain} is not a polymer chain of this PDB")

        chain_type = antibody.get(chain)

        if chain_type in ((HEAVY_TYPE,) if col == "Hchain" else LIGHT_TYPES):
            continue

        if chain_type is None:
            if custom_chain_mode:
                log.info(f"{col}={chain} is not an antibody chain, numbered as given")
                continue
            raise UnusableChains(f"{col}={chain} is not an antibody chain. {AS_GIVEN}")

        reads_as = "Hchain" if chain_type == HEAVY_TYPE else "Lchain"
        raise UnusableChains(
            f"{col}={chain} is not {CHAIN_NAMES[col]}: ANARCII reads it as "
            f"{CHAIN_NAMES[reads_as]}. {AS_GIVEN}"
        )


def _reconcile_chain_columns(
    pdb, pdbs_csv, i, numbered, antibody, chains_specified, custom_chain_mode
):
    """Checks the Hchain/Lchain columns against ANARCII, or corrects them

    Chains the user named are a claim, so they are only checked. Chains AntiFold
    guessed from file order are rewritten to ANARCII's reading. In custom_chain_mode
    the columns are a selection, so unnamed ones are left exactly as given.
    """

    named = _candidate_chains(pdbs_csv, i) if chains_specified else {}

    if named:
        _check_named_chains(named, numbered, antibody, custom_chain_mode)
        return

    if custom_chain_mode:
        return

    assigned = _assign_chains(antibody)

    if "Lchain" not in assigned:
        raise UnusableChains(
            f"only a heavy chain ({assigned['Hchain']}) found, and an unpaired chain "
            f"needs --nanobody_chain or --custom_chain_mode"
        )

    log.info(f"{pdb}: ANARCII found {_format_chains(assigned)}")

    for col in CHAIN_COLUMNS:
        if col in pdbs_csv.columns:
            pdbs_csv.loc[i, col] = None
    for col, chain in assigned.items():
        pdbs_csv.loc[i, col] = chain


def _write_structure(structure, out_dir, pdb_path):
    """Writes the renumbered structure out in the format it came in"""

    # AntiFold reads the first model only, so the rest would be written out
    # still carrying their original numbering
    while len(structure) > 1:
        del structure[1]

    out_path = f"{out_dir}/{os.path.basename(pdb_path)}"

    if out_path.endswith(".cif"):
        structure.make_mmcif_document().write_file(out_path)
    else:
        structure.write_pdb(out_path)


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

    Every chain ANARCII reads as an antibody chain is renumbered; antigens and
    anything else keep the numbering they came with. The Hchain/Lchain columns are
    then checked or corrected - see _reconcile_chain_columns.
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
            numbered = _number_chains(model, structure, pdb_path)
            antibody = _antibody_chains(numbered)
            _reconcile_chain_columns(
                _pdb, pdbs_csv, i, numbered, antibody,
                chains_specified, custom_chain_mode,
            )
        except UnusableChains as e:
            log.warning(f"WARNING: skipping {_pdb}: {e}")
            skipped.append(str(_pdb))
            continue

        for chain, chain_type in antibody.items():
            renumber_pdbx(structure, 0, chain, numbered[chain])
            log.info(f"{_pdb} chain {chain}: IMGT renumbered ({chain_type})")

        _write_structure(structure, out_dir, pdb_path)
        kept.append(i)

    if skipped:
        log.warning(
            f"WARNING: skipped {len(skipped)}/{len(pdbs_csv)} PDBs on chain "
            f"mismatch: {', '.join(skipped)}"
        )

    if not kept:
        raise ValueError(
            f"No PDBs left to run: ANARCII could not use the chains of any of the "
            f"{len(pdbs_csv)} given. Re-run with --skip_anarcii_numbering to use "
            f"the chains and numbering as given"
        )

    # Reset index: InverseData indexes rows by position, via df.loc[i] over range(len(df))
    return pdbs_csv.loc[kept].reset_index(drop=True), out_dir


def warn_if_not_imgt_numbered(pdbs_csv, pdb_dir, custom_chain_mode=False):
    """Warns for each PDB whose heavy chain is not IMGT numbered

    Renumbering is off, so region masks are read off the input numbering as given.
    In custom_chain_mode the first chain need not be a heavy chain, so there is
    nothing to check it against.
    """

    if custom_chain_mode:
        return

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
                f"be wrong. Drop --skip_anarcii_numbering to renumber it with ANARCII"
            )
