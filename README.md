AntiFold
==============================

Code for AntiFold paper (unpublished)

Webserver: [OPIG webserver](https://opig.stats.ox.ac.uk/webapps/AntiFold/)

Code: [antifold_code.zip](https://opig.stats.ox.ac.uk/data/downloads/AntiFold/antifold_code.zip)

Model: [model.pt](https://opig.stats.ox.ac.uk/data/downloads/AntiFold/models/model.pt)

License: [BSD 3-Clause](https://opig.stats.ox.ac.uk/data/downloads/AntiFold/LICENSE)

# Colab
To test the method out without installing it you can try this: [![Open In Colab](images/colab-badge.svg)](https://colab.research.google.com/drive/1TTfgjoZx3mzF5u4e9b4Un9Y7b_rqXc_4)

## Install and run AntiFold

Use the <a href="https://opig.stats.ox.ac.uk/data/downloads/AntiFold/notebook.ipynb">Jupyter notebook</a> or follow this guide with example PDBs:

### Download AntiFold
```bash
# Download code
mkdir -p antifold_code && cd antifold_code
wget https://opig.stats.ox.ac.uk/data/downloads/AntiFold/antifold_code.zip
unzip antifold_code.zip

# Download model
mkdir -p models
wget -P models/ https://opig.stats.ox.ac.uk/data/downloads/AntiFold/models/model.pt
```

### Install AntiFold (CPU)
```bash
conda create --name antifold python=3.10 -y
conda activate antifold
conda install -c pytorch pytorch
conda install -c pyg pyg -y
conda install -c conda-forge pip -y
# Install AntiFold from antifold_code directory
pip install .
pip install torch-geometric==2.4.0
```

### Install AntiFold (GPU)
```bash
conda create --name antifold python=3.10 -y
conda activate antifold
conda install -c conda-forge pytorch-gpu
conda install -c pyg pyg -y
conda install -c conda-forge pip -y
# Install AntiFold from antifold_code directory
pip install .
pip install torch-geometric==2.4.0
```

### Run AntiFold (residue probabilities, embeddings, sampled sequences)
```bash
# Residue probabilities, sample 10 sequences in CDRH3 from single PDB
python antifold/main.py \
    --out_dir output/single_pdb \
    --pdb_file data/pdbs/6y1l_imgt.pdb \
    --heavy_chain H \
    --light_chain L \
    --num_seq_per_target 10 \
    --sampling_temp "0.2" \
    --regions "CDRH3"

# Residue probabilities, sample 10 sequences at temperatures 0.20 & 0.30 in CDRs 1-3 from folder of PDBs
python antifold/main.py \
    --out_dir output/example_pdbs \
    --pdbs_csv data/example_pdbs.csv \
    --pdb_dir data/pdbs \
    --num_seq_per_target 10 \
    --sampling_temp "0.20 0.30" \
    --regions "CDR1 CDR2 CDR3"

# Extract ESM-IF1 embeddings from PDBs in folder, from all chains listed in untested.csv (not limited to VH/VL)
python antifold/main.py \
    --out_dir output/untested/ \
    --pdbs_csv data/untested.csv \
    --pdb_dir data/untested/ \
    --use_esm_if1_weights \
    --custom_chain_mode \
    --extract_embeddings
```

## Input parameters
Required parameters:
```text
- Option 1: PDB file (--pdb_file) + specified heavy and light chain (--heavy_chain and --light_chain)
- Option 2: PDB folder (--pdb_dir) + CSV file specifying chains (--pdbs_csv)
- Output directory (--out_dir) for output inverse folding probabilities CSV and optional generated sequences FASTA
```

Parameters for generating new sequences:
```text
- Number of sequences to generate (--num_seq_per_target)
- Region to mutate (--region) based on inverse folding probabilities. Select from list in IMGT_dict (e.g. 'CDRH1 CDRH2 CDRH3')
- Sampling temperature (--sampling_temp) controls generated sequence diversity, by scaling the inverse folding probabilities before sampling. Temperature = 1 means no change, while temperature ~ 0 only samples the most likely amino-acid at each position (acts as argmax).
```

Optional parameters:
```text
- Multi-chain mode for including antigen or other chains (--custom_chain_mode) - Nb, experimental (see example above)
- Extract latent representations of PDB within model (--extract_embeddings)
- Use ESM-IF1 instead of AntiFold model weights (--model_path "ESM-IF1")
```

## Example output
Output CSV with residue log-probabilities: Residue probabilities: <a href="https://opig.stats.ox.ac.uk/data/downloads/AntiFold/output/example_pdbs/6y1l_imgt.csv">6y1l_imgt.csv</a>
- pdb_pos - PDB residue number
- pdb_chain - PDB chain
- aa_orig - PDB residue (e.g. 112)
- aa_pred - Top predicted residue by AntiFold (argmax) for this position
- pdb_posins - PDB residue number with insertion code (e.g. 112A)
- perplexity - Inverse folding tolerance (higher is more tolerant) to mutations. See paper for more details.
- Amino-acids - Inverse folding scores (log-likelihood) for the given position
```csv
pdb_pos,pdb_chain,aa_orig,aa_pred,pdb_posins,perplexity,A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y
2,H,V,M,2,1.6488,-4.9963,-6.6117,-6.3181,-6.3243,-6.7570,-4.2518,-6.7514,-5.2540,-6.8067,-5.8619,-0.0904,-6.5493,-4.8639,-6.6316,-6.3084,-5.1900,-5.0988,-3.7295,-8.0480,-7.3236
3,H,Q,Q,3,1.3889,-10.5258,-12.8463,-8.4800,-4.7630,-12.9094,-11.0924,-5.6136,-10.9870,-3.1119,-8.1113,-9.4382,-6.2246,-13.3660,-0.0701,-4.9957,-10.0301,-6.8618,-7.5810,-13.6721,-11.4157
4,H,L,L,4,1.0021,-13.3581,-12.6206,-17.5484,-12.4801,-9.8792,-13.6382,-14.8609,-13.9344,-16.4080,-0.0002,-9.2727,-16.6532,-14.0476,-12.5943,-15.4559,-16.9103,-17.0809,-10.5670,-13.5334,-13.4324
...
```

Output FASTA file with sampled sequences: <a href="https://opig.stats.ox.ac.uk/data/downloads/AntiFold/output/example_pdbs/6y1l_imgt.fasta">6y1l_imgt.fasta</a>
- T: Temperature used for design
- score: average log-odds of residues in the sampled region
- global_score: average log-odds of all residues (IMGT positions 1-128)
- regions: regions selected for design
- seq_recovery: # mutations / total sequence length
- mutations: # mutations from original PDB sequence
```fasta
>6y1l_imgt , score=0.2934, global_score=0.2934, regions=['CDR1', 'CDR2', 'CDRH3'], model_name=AntiFold, seed=42
VQLQESGPGLVKPSETLSLTCAVSGYSISSGYYWGWIRQPPGKGLEWIGSIYHSGSTYYN
PSLKSRVTISVDTSKNQFSLKLSSVTAADTAVYYCAGLTQSSHNDANWGQGTLVTVSS/V
LTQPPSVSAAPGQKVTISCSGSSSNIGNNYVSWYQQLPGTAPKRLIYDNNKRPSGIPDRF
SGSKSGTSATLGITGLQTGDEADYYCGTWDSSLNPVFGGGTKLEIKR
> T=0.20, sample=1, score=0.3930, global_score=0.1869, seq_recovery=0.8983, mutations=12
VQLQESGPGLVKPSETLSLTCAVSGASITSSYYWGWIRQPPGKGLEWIGSIYYSGSTYYN
PSLKSRVTISVDTSKNQFSLKLSSVTAADTAVYYCAGLYGSPWSNPYWGQGTLVTVSS/V
LTQPPSVSAAPGQKVTISCSGSSSNIGNNYVSWYQQLPGTAPKRLIYDNNKRPSGIPDRF
SGSKSGTSATLGITGLQTGDEADYYCGTWDSSLNPVFGGGTKLEIKR
...
```
## Example notebook
Notebook: <a href="https://opig.stats.ox.ac.uk/data/downloads/AntiFold/notebook.ipynb">notebook.ipynb</a>

```python
import pandas as pd

# Put IMGT numbered PDBs (Fv only, IMGT position 1-128) to process and load a CSV file with PDB names and heavy/light chains
# Define the PDB and chains in DataFrame
pdb_dir = "data/pdbs"
df_pdbs = pd.read_csv("data/example_pdbs.csv")

# Regions to mutate (IMGT)
regions_to_mutate = ["CDR1", "CDR2", "CDR3H"]

# Load model
import antifold.main as antifold
model = antifold.load_IF1_model("models/model.pt")

# Sample from PDBs, 10 sequences each at temperature 0.50 in regions CDR1, CDR2, CDR3H
pdb_output_dict = antifold.sample_pdbs(
                    model,
                    pdbs_csv_or_dataframe=df_pdbs, # Path to CSV file, or a DataFrame
                    regions_to_mutate=regions_to_mutate,
                    pdb_dir="data/pdbs",
                    sample_n=10,
                    sampling_temp=0.50,
                    limit_expected_variation=False
                    )

# Output dictionary with sequences, and residue probabilities or log-odds
pdbs = pdb_output_dict.keys()

# Residue log probabilities
df_logprobs = pdb_output_dict["6y1l_imgt"]["logprobs"]

# Sampled sequences
fasta_dict = pdb_output_dict["6y1l_imgt"]["sequences"]
```

## Usage
```bash
usage: 
    # Predict on example PDBs in folder
    python antifold/main.py     --pdbs_csv data/example_pdbs.csv     --pdb_dir data/pdbs     --out_dir output/

Predict inverse folding probabilities for antibody variable domain, and sample sequences with maintained fold.
PDB structures should be IMGT-numbered, paired heavy and light chain variable domains (positions 1-128).

For IMGT numbering PDBs use SAbDab or https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabpred/anarci/

options:
  -h, --help            show this help message and exit
  --pdb_file PDB_FILE   Input PDB file (for single PDB predictions)
  --heavy_chain HEAVY_CHAIN
                        Ab heavy chain (for single PDB predictions)
  --light_chain LIGHT_CHAIN
                        Ab light chain (for single PDB predictions)
  --antigen_chain ANTIGEN_CHAIN
                        Antigen chain (experimental)
  --pdbs_csv PDBS_CSV   Input CSV file with PDB names and H/L chains (multi-PDB predictions)
  --pdb_dir PDB_DIR     Directory with input PDB files (multi-PDB predictions)
  --out_dir OUT_DIR     Output directory
  --regions REGIONS     Space-separated regions to mutate. Default 'CDR1 CDR2 CDR3H'
  --num_seq_per_target NUM_SEQ_PER_TARGET
                        Number of sequences to sample from each antibody PDB (default 0)
  --sampling_temp SAMPLING_TEMP
                        A string of temperatures e.g. '0.20 0.25 0.50' (default 0.20). Sampling temperature for amino acids. Suggested values 0.10, 0.15, 0.20, 0.25, 0.30. Higher values will lead to more diversity.
  --limit_variation     Limit variation to as many mutations as expected from temperature sampling
  --extract_embeddings  Extract per-residue embeddings from AntiFold / ESM-IF1
  --custom_chain_mode   Custom chain input (experimental, e.g. single chain, inclusion of antigen chain or any chains with ESM-IF1)
  --exclude_heavy       Exclude heavy chain from sampling
  --exclude_light       Exclude light chain from sampling
  --batch_size BATCH_SIZE
                        Batch-size to use
  --num_threads NUM_THREADS
                        Number of CPU threads to use for parallel processing (0 = all available)
  --seed SEED           Seed for reproducibility
  --model_path MODEL_PATH
                        AntiFold model weights. See --use_esm_if1_weights flag to use ESM-IF1 weights instead of AntiFold
  --use_esm_if1_weights
                        Use ESM-IF1 weights instead of AntiFold
  --verbose VERBOSE     Verbose printing
```

## IMGT regions dict
Used to specify which regions to mutate in an IMGT numbered PDB
- IMGT numbered PDBs: [https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab)
- Renumber existing PDBs with ANARCI: [https://github.com/oxpig/ANARCI](https://github.com/oxpig/ANARCI)
- Read more: [https://www.imgt.org/IMGTScientificChart/Numbering/IMGTIGVLsuperfamily.html](https://www.imgt.org/IMGTScientificChart/Numbering/IMGTIGVLsuperfamily.html)

```python
IMGT_dict = {
    "all": range(1, 128 + 1),
    "allH": range(1, 128 + 1),
    "allL": range(1, 128 + 1),
    "FWH": list(range(1, 26 + 1)) + list(range(40, 55 + 1)) + list(range(66, 104 + 1)),
    "FWL": list(range(1, 26 + 1)) + list(range(40, 55 + 1)) + list(range(66, 104 + 1)),
    "CDRH": list(range(27, 39)) + list(range(56, 65 + 1)) + list(range(105, 117 + 1)),
    "CDRL": list(range(27, 39)) + list(range(56, 65 + 1)) + list(range(105, 117 + 1)),
    "FW1": range(1, 26 + 1),
    "FWH1": range(1, 26 + 1),
    "FWL1": range(1, 26 + 1),
    "CDR1": range(27, 39),
    "CDRH1": range(27, 39),
    "CDRL1": range(27, 39),
    "FW2": range(40, 55 + 1),
    "FWH2": range(40, 55 + 1),
    "FWL2": range(40, 55 + 1),
    "CDR2": range(56, 65 + 1),
    "CDRH2": range(56, 65 + 1),
    "CDRL2": range(56, 65 + 1),
    "FW3": range(66, 104 + 1),
    "FWH3": range(66, 104 + 1),
    "FWL3": range(66, 104 + 1),
    "CDR3": range(105, 117 + 1),
    "CDRH3": range(105, 117 + 1),
    "CDRL3": range(105, 117 + 1),
    "FW4": range(118, 128 + 1),
    "FWH4": range(118, 128 + 1),
    "FWL4": range(118, 128 + 1),
}
```
