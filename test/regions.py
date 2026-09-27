"""--regions accepts IMGT region names and chain-prefixed IMGT positions.

IMGT_dict defines each region as a position range, so a region name and the
positions it spans must sample identically at a fixed seed.
"""

import subprocess
import sys

import pandas as pd

from antifold.antiscripts import get_imgt_mask

OUT_DIR = "output/regions"
PAIRED = ["--pdb_file", "data/pdbs/6y1l_imgt.pdb", "--heavy_chain", "H", "--light_chain", "L"]
NANOBODY = ["--pdb_file", "data/nanobody/8oi2_imgt.pdb", "--nanobody_chain", "B"]
PAIRED_FASTA = f"{OUT_DIR}/6y1l_imgt_HL.fasta"
NANOBODY_FASTA = f"{OUT_DIR}/8oi2_imgt_B.fasta"


def run_antifold(pdb_args, regions, extra_args=()):
    return subprocess.run(
        [sys.executable, "antifold/main.py", *pdb_args, *extra_args,
         "--regions", regions, "--out_dir", OUT_DIR],
        capture_output=True,
    )


def sample_seqs(pdb_args, regions, fasta):
    run_antifold(
        pdb_args, regions,
        ["--num_seq_per_target", "10", "--sampling_temp", "0.30", "--seed", "42"],
    ).check_returncode()

    with open(fasta) as f:
        return [line for line in f if not line.startswith(">")]


def assert_same(pdb_args, regions_a, regions_b, fasta):
    if sample_seqs(pdb_args, regions_a, fasta) != sample_seqs(pdb_args, regions_b, fasta):
        sys.exit(f"FAIL: '{regions_a}' and '{regions_b}' sampled different sequences")
    print(f"OK: '{regions_a}' == '{regions_b}'")


def assert_differs(pdb_args, regions_a, regions_b, fasta):
    if sample_seqs(pdb_args, regions_a, fasta) == sample_seqs(pdb_args, regions_b, fasta):
        sys.exit(f"FAIL: '{regions_a}' and '{regions_b}' sampled identical sequences")
    print(f"OK: '{regions_a}' != '{regions_b}'")


print("\n### Region names vs the positions they span ###")
assert_same(PAIRED, "CDRH1", "H:27-38", PAIRED_FASTA)
assert_same(PAIRED, "CDRL1", "L:27-38", PAIRED_FASTA)
assert_same(PAIRED, "CDRH1 CDRH2", "CDRH1 H:56-65", PAIRED_FASTA)
assert_same(NANOBODY, "CDRH1", "H:27-38", NANOBODY_FASTA)

print("\n### Positions stay on the chain they name ###")
assert_differs(PAIRED, "H:27-38", "CDR1", PAIRED_FASTA)

# Positions spanning more than one region have no equivalent name to compare against,
# and antigen residues (no assumed_region) must never be selected
print("\n### Masking selects exactly the positions asked for ###")
df = pd.DataFrame({
    "pdb_chain":      ["H"] * 6 + ["L"] * 4 + ["Y"] * 2,
    "pdb_pos":        [10, 11, 12, 45, 111, 120] + [5, 66, 70, 99] + [10, 66],
    "assumed_region": ["FWH1", "FWH1", "FWH1", "FWH2", "CDRH3", "FWH4"]
                      + ["FWL1", "FWL3", "FWL3", "FWL3"] + ["", ""],
})

expectations = {
    ("H:10-12,111", "L:66-70"): {("H", 10), ("H", 11), ("H", 12), ("H", 111), ("L", 66), ("L", 70)},
    ("H:10-12",): {("H", 10), ("H", 11), ("H", 12)},
    ("FWH",): {("H", 10), ("H", 11), ("H", 12), ("H", 45), ("H", 120)},
    ("FWL",): {("L", 5), ("L", 66), ("L", 70), ("L", 99)},
    ("CDR3", "L:5"): {("H", 111), ("L", 5)},
}

for regions, expected in expectations.items():
    mask = get_imgt_mask(df, list(regions))
    selected = set(zip(df["pdb_chain"][mask], df["pdb_pos"][mask]))
    if selected != expected:
        sys.exit(f"FAIL: {list(regions)} selected {sorted(selected)}, expected {sorted(expected)}")
    print(f"OK: {list(regions)} -> {sorted(expected)}")

# "CDR1  CDR2" (double space) used to select every residue with no IMGT region,
# which on an antibody-antigen complex means designing the antigen
print("\n### Ambiguous or malformed regions are rejected ###")
for bad_regions in ["10-12", "111", "H:12-10", "H:", "H:10-", "CDR1  CDR2", "", "CDR4", "FWH9"]:
    if run_antifold(PAIRED, bad_regions, ["--num_seq_per_target", "2"]).returncode == 0:
        sys.exit(f"FAIL: --regions '{bad_regions}' was accepted")
    print(f"OK: --regions '{bad_regions}' rejected")

print("\nAll region tests passed")
