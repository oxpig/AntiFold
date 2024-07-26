#!/bin/bash
set -e
set -o nounset
set -o pipefail
if [[ ${TRACE-0} == "1" ]]; then
  set -o xtrace
fi

NUM_TESTS=7

# Run AntiFold on single PDB/CIF file
# Nb: Assumes first chain heavy, second chain light
echo -e "\n### Test 1 / $NUM_TESTS ###"
python antifold/main.py \
    --pdb_file data/pdbs/6y1l_imgt.pdb

# Antibody-antigen complex
echo -e "\n### Test 2 / $NUM_TESTS ###"
python antifold/main.py \
    --pdb_file data/antibody_antigen/3hfm.pdb \
    --heavy_chain H \
    --light_chain L \
    --antigen_chain Y

# Nanobody or single-chain
echo -e "\n### Test 3 / $NUM_TESTS ###"
python antifold/main.py \
    --pdb_file data/nanobody/8oi2_imgt.pdb \
    --nanobody_chain B

# Folder of PDB/CIFs
# Nb: Assumes first chain heavy, second light
echo -e "\n### Test 4 / $NUM_TESTS ###"
python antifold/main.py \
    --pdb_dir data/pdbs

# Specify chains to run in a CSV file (e.g. antibody-antigen complex)
echo -e "\n### Test 5 / $NUM_TESTS ###"
python antifold/main.py \
    --pdb_dir data/antibody_antigen \
    --pdbs_csv data/antibody_antigen.csv

# Sample sequences 10x
echo -e "\n### Test 6 / $NUM_TESTS ###"
python antifold/main.py \
    --pdb_file data/pdbs/6y1l_imgt.pdb \
    --heavy_chain H \
    --light_chain L \
    --num_seq_per_target 10 \
    --sampling_temp "0.2" \
    --regions "CDR1 CDR2 CDR3"

# Run all chains with ESM-IF1 model weights
echo -e "\n### Test 7 / $NUM_TESTS ###"
python antifold/main.py \
    --pdb_dir data/pdbs \
    --esm_if1_mode