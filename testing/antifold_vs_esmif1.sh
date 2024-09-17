echo "Using weights AntiFold ..."
python antifold/main.py \
    --pdb_file data/pdbs/6y1l_imgt.pdb \
    --out_dir output/antifold

echo "Using weights ESM-IF1 ..."
python antifold/main.py \
    --pdb_file data/pdbs/6y1l_imgt.pdb \
    --esm_if1_mode \
    --out_dir output/esmif1