python antifold/main.py \
    --pdb_file data/nanobody/nanobody_antigen_9hzj_imgt.pdb \
    --heavy_chain A \
    --antigen_chain B \
    --nanobody_mode \
    --num_seq_per_target 10 \
    --sampling_temp "0.2" \
    --regions "CDRH"