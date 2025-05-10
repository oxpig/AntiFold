python antifold/main.py \
    --pdb_file data/antibody_antigen/3hfm.pdb \
    --heavy_chain H \
    --light_chain L \
    --antigen_chain Y \
    --num_seq_per_target 10 \
    --sampling_temp "0.2" \
    --regions "CDR1 CDR2 CDR3"
