import glob
import logging
import os
import sys
import warnings
import urllib.request
from pathlib import Path

ROOT_PATH = Path(os.path.dirname(__file__)).parent
sys.path.insert(0, ROOT_PATH)

import re
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from Bio import SeqIO
from Bio.Seq import Seq
from biotite.structure.io import pdb

import antifold.esm
from antifold.esm_util_custom import CoordBatchConverter_mask_gpu
from antifold.if1_dataset import InverseData

log = logging.getLogger(__name__)

amino_list = list("ACDEFGHIKLMNPQRSTVWY")

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


def extract_chains_biotite(pdb_file):
    """Extract chains in order"""
    pdbf = pdb.PDBFile.read(pdb_file)
    structure = pdb.get_structure(pdbf, model=1)
    return pd.unique(structure.chain_id)


def generate_pdbs_csv(pdbs_dir, max_chains=10):
    """Generates AntiFOld CSV with PDB/CIF basenames + chains (up to 10)"""

    columns = [
        "pdb",
        "Hchain",
        "Lchain",
        "chain3",
        "chain4",
        "chain5",
        "chain6",
        "chain7",
        "chain8",
        "chain9",
        "chain10",
    ]
    pdbs_csv = pd.DataFrame(columns=columns)
    pdbs = glob.glob(f"{pdbs_dir}/*.[pdb cif]*")

    for i, pdb_file in enumerate(pdbs):
        _pdb = os.path.splitext(os.path.basename(pdb_file))[0]

        # Extract first 10 chains
        chains = extract_chains_biotite(pdb_file)[:max_chains]

        # Put pdb and chains in dataframe
        row = [_pdb] + list(chains)
        _cols = columns[: len(row)]
        pdbs_csv.loc[i, _cols] = row

    return pdbs_csv


def load_IF1_checkpoint(model, checkpoint_path: str = ""):
    # Load
    log.debug(f"Loading AntiFold model {checkpoint_path} ...")

    # Check for CPU/GPU load
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dict = torch.load(checkpoint_path, map_location=torch.device(device))

    # PYL checkpoint ?
    if "model_state_dict" in checkpoint_dict.keys():
        pretrained_dict = {
            re.sub("model.", "", k): v
            for k, v in checkpoint_dict["model_state_dict"].items()
        }

    # PYL checkpoint ?
    elif "state_dict" in checkpoint_dict.keys():
        # Remove "model." from keys
        pretrained_dict = {
            re.sub("model.", "", k): v for k, v in checkpoint_dict["state_dict"].items()
        }

    # IF1-raw?
    else:
        pretrained_dict = checkpoint_dict

    # Load pretrained weights
    model.load_state_dict(pretrained_dict)

    return model


def load_model(checkpoint_path: str = ""):
    """Load raw/FT IF1 model"""

    # Check that AntiFold weights are downloaded
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = f"{root_dir}/models/model.pt"

    if not os.path.exists(model_path):
        log.warning(
            f"Downloading AntiFold model weights from https://opig.stats.ox.ac.uk/data/downloads/AntiFold/models/model.pt to {model_path}"
        )
        url = "https://opig.stats.ox.ac.uk/data/downloads/AntiFold/models/model.pt"
        filename = model_path

        os.makedirs(f"{root_dir}/models", exist_ok=True)
        urllib.request.urlretrieve(url, filename)

    if not os.path.exists(model_path) and not checkpoint_path == "ESM-IF1":
        raise Exception(
            f"Unable to find model weights. File does not exist: {checkpoint_path}"
        )

    # Download IF1 weights
    if checkpoint_path == "ESM-IF1":
        log.info(
            f"NOTE: Loading ESM-IF1 weights instead of fine-tuned AntiFold weights"
        )
        # Suppress regression weights warning - not needed
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model, _ = antifold.esm.pretrained.esm_if1_gvp4_t16_142M_UR50()

    # Load AntiFold weights locally
    else:
        model, _ = antifold.esm.pretrained._load_IF1_local()
        model = load_IF1_checkpoint(model, model_path)

    # Evaluation mode when predicting
    model = model.eval()

    # Send to CPU/GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ = model.to(device)
    log.info(f"Loaded model to {device}.")

    return model


def get_dataset_pdb_name_chainsname_res_posins_chains(dataset, idx):
    """Gets PDB sequence, position+insertion codes and chains from dataset"""

    # PDB path
    pdb_path = dataset.pdb_info_dict[idx]["pdb_path"]

    # PDB names
    pdb_name = dataset.pdb_info_dict[idx]["pdb"]
    pdb_chainsname = dataset.pdb_info_dict[idx]["pdb_chainsname"]

    # Sequence - gaps
    seq = dataset[idx][2]
    pdb_res = [aa for aa in seq if aa != "-"]

    # Position + insertion codes - gaps
    posinschain = dataset[idx][4]
    posinschain = [p for p in posinschain if p != "nan"]

    # Extract position + insertion code + chain (1-letter)
    pdb_posins = [p[:-1] for p in posinschain]
    pdb_chains = [p[-1] for p in posinschain]

    return pdb_name, pdb_chainsname, pdb_res, pdb_posins, pdb_chains


def logits_to_seqprobs_list(logits, tokens):
    """Convert logits (bs x 35 x L) ot list of L x 20 seqprobs"""

    alphabet = antifold.esm.data.Alphabet.from_architecture("invariant_gvp")

    mask_gap = tokens[:, 1:] != 30  # 30 is gap
    mask_pad = tokens[:, 1:] != alphabet.padding_idx  # 1 is padding
    mask_combined = mask_gap & mask_pad

    # Check that only 10x gap ("-") per sequence!
    # batch_size = logits.shape[0]
    # assert (mask_gap == False).sum() == batch_size * 10

    # Filter out gap (-) and padding (<pad>) positions, only keep 21x amino-acid probs (4:24) + "X"
    seqprobs_list = [logits[i, 4:25, mask_combined[i]] for i in range(len(logits))]

    return seqprobs_list


def get_dataset_dataloader(
    pdbs_csv_or_dataframe, pdb_dir, batch_size, custom_chain_mode=False, num_threads=0
):
    """Prepares dataset/dataoader from CSV file containing PDB paths and H/L chains"""

    # Set number of threads & workers
    if num_threads >= 1:
        torch.set_num_threads(num_threads)
        num_threads = min(num_threads, 4)

    # Load PDB coordinates
    dataset = InverseData(
        gaussian_noise_flag=False, custom_chain_mode=custom_chain_mode,
    )
    dataset.populate(pdbs_csv_or_dataframe, pdb_dir)

    # Prepare torch dataloader at specified batch size
    alphabet = antifold.esm.data.Alphabet.from_architecture("invariant_gvp")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=CoordBatchConverter_mask_gpu(alphabet),
        num_workers=num_threads,
    )

    return dataset, dataloader


def dataset_dataloader_to_predictions_list(
    model, dataset, dataloader, batch_size=1, extract_embeddings=False
):
    """Get PDB predictions from a dataloader"""

    # Check that dataloader and dataset match, and no random shuffling
    if "random" in str(dataloader.sampler).lower():
        raise ValueError(
            "Torch DataLoader sampler must not be random. Did you forget to set torch.utils.data.DataLoader ... shuffle=False?"
        )
    if dataloader.dataset is not dataset:
        raise ValueError("Dataloader and dataset must match to align samples!")

    # Get all batch predictions
    all_seqprobs_list = []
    all_embeddings_list = []
    for bi, batch in enumerate(dataloader):
        start_index = bi * batch_size
        end_index = min(
            start_index + batch_size, len(dataset)
        )  # Adjust for the last batch
        log.info(
            f"Predicting batch {bi+1}/{len(dataloader)}: PDBs {start_index+1}-{end_index} out of {len(dataset)} total"
        )  # -1 because the end_index is exclusive

        # Test dataloader
        (
            coords,
            confidence,
            strs,
            tokens,
            padding_mask,
            loss_masks,
            res_pos,
            posinschain_list,
            targets,
        ) = batch

        # Test forward
        with torch.no_grad():
            prev_output_tokens = tokens[:, :-1]
            logits, extra = model.forward(  # bs x 35 x seq_len, exlude bos, eos
                coords,
                padding_mask,  # Includes masked positions
                confidence,
                prev_output_tokens,
                features_only=False,
            )

            logits = logits.detach().cpu().numpy()
            tokens = tokens.detach().cpu().numpy()

            # List of L x 21 seqprobs (20x AA, 21st == "X")
            L = logits_to_seqprobs_list(logits, tokens)
            all_seqprobs_list.extend(L)

            if extract_embeddings:
                # Extract embeddings from batch
                padding_mask = padding_mask.detach().cpu().numpy()  # bs x len
                Es = extra["inner_states"][0].detach().cpu().numpy()  # len x bs x 512
                # L[0] = encoder output
                # L[1] = decoder input
                # L[2] = decoder output layer 1
                # L[3] = decoder output layer 2 ...
                Es = np.transpose(Es, (1, 0, 2))  # bs x len x 512

                # Embeddings without padding and bos/eos
                for e, pad in zip(Es, padding_mask):
                    # e: len x 512
                    e = e[~pad]  # (len - pad) x 512
                    e = e[1:-1]  # seq_len x 512
                    all_embeddings_list.append(e)  # seq_len x 512

    return all_seqprobs_list, all_embeddings_list


def predictions_list_to_df_logits_list(all_seqprobs_list, dataset, dataloader):
    """PDB preds list to PDB DataFrames"""

    # Check that dataloader and dataset match, and no random shuffling
    if "random" in str(dataloader.sampler).lower():
        raise ValueError(
            "Torch DataLoader sampler must not be random. Did you forget to set torch.utils.data.DataLoader ... shuffle=False?"
        )
    if dataloader.dataset is not dataset:
        raise ValueError("Dataloader and dataset must match to align samples!")

    # Create DataFrame for each PDB
    all_df_logits_list = []
    for idx, seq_probs in enumerate(all_seqprobs_list):
        # Get PDB sequence, position+insertion code and H+L chain idxs
        (
            pdb_name,
            pdb_chainsname,
            pdb_res,
            pdb_posins,
            pdb_chains,
        ) = get_dataset_pdb_name_chainsname_res_posins_chains(dataset, idx)

        # Check matches w/ residue probs
        assert len(seq_probs) == len(pdb_posins)

        # Logits to DataFrame
        alphabet = antifold.esm.data.Alphabet.from_architecture("invariant_gvp")
        df_logits = pd.DataFrame(data=seq_probs, columns=alphabet.all_toks[4:25],)

        # Limit to 20x amino-acids probs
        _alphabet = list("ACDEFGHIKLMNPQRSTVWY")
        df_logits = df_logits[_alphabet]

        # Add PDB info
        positions = pdb_posins_to_pos(pd.Series(pdb_posins))
        top_res = np.array((_alphabet))[df_logits[_alphabet].values.argmax(axis=1)]
        perplexity = calc_pos_perplexity(df_logits)

        # Add to DataFrame
        df_logits.name = pdb_chainsname
        df_logits.insert(0, "pdb_posins", pdb_posins)
        df_logits.insert(1, "pdb_chain", pdb_chains)
        df_logits.insert(2, "pdb_res", pdb_res)
        df_logits.insert(3, "top_res", top_res)
        df_logits.insert(4, "pdb_pos", positions)
        df_logits.insert(5, "perplexity", perplexity)

        # Skip if not IMGT numbered - 10 never found in H-chain IMGT numbered PDBs
        Hchain = pdb_chains[0]
        Hpos = positions[pdb_chains == Hchain]
        if 10 in Hpos and not dataset.custom_chain_mode:
            log.error(
                f"WARNING: PDB {pdb_name} seems to not be IMGT numbered! Output probabilities may be affected. See https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabpred/anarci/"
            )
        # Limit to IMGT positions only (only ones trained on)
        # imgt_mask = get_imgt_mask(df_logits, imgt_regions=["all"])
        # df_logits = df_logits[imgt_mask]

        all_df_logits_list.append(df_logits)

    return all_df_logits_list


def df_logits_list_to_logprob_csvs(
    df_logits_list, out_dir, embeddings_list=False, float_format="%.4f"
):
    """Save df_logits_list to CSVs"""
    os.makedirs(out_dir, exist_ok=True)
    log.info(f"Saving {len(df_logits_list)} CSVs to {out_dir}")

    for i, df in enumerate(df_logits_list):
        # Convert to log-probs
        df_out = df_logits_to_logprobs(df)
        # Save
        outpath = f"{out_dir}/{df.name}.csv"
        log.info(f"Writing {df.name} per-residue log probs CSV to {outpath}")
        df_out.to_csv(outpath, float_format=float_format, index=False)

        if embeddings_list:
            # Save embeddingsl
            outpath = f"{out_dir}/{df.name}.npy"
            log.info(f"Writing {df.name} per-residue embeddings to {outpath}")
            np.save(outpath, embeddings_list[i])


def seed_everything(seed: int):
    # https://gist.github.com/ihoromi4/b681a9088f348942b01711f251e5f964
    import os
    import random

    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_pdbs_logits(
    model,
    pdbs_csv_or_dataframe,
    pdb_dir,
    out_dir=False,
    batch_size=1,
    extract_embeddings=False,
    custom_chain_mode=False,
    num_threads=0,
    save_flag=False,
    float_format="%.4f",
    seed=42,
):
    """Predict PDBs from a CSV file"""

    # if save_flag:
    #    log.info(f"Saving prediction CSVs to {out_dir}")

    seed_everything(seed)

    # Load PDBs
    dataset, dataloader = get_dataset_dataloader(
        pdbs_csv_or_dataframe,
        pdb_dir,
        batch_size=batch_size,
        custom_chain_mode=custom_chain_mode,
        num_threads=num_threads,
    )

    # Predict PDBs -> df_logits
    predictions_list, embeddings_list = dataset_dataloader_to_predictions_list(
        model,
        dataset,
        dataloader,
        batch_size=batch_size,
        extract_embeddings=extract_embeddings,
    )
    df_logits_list = predictions_list_to_df_logits_list(
        predictions_list, dataset, dataloader
    )

    # Save df_logits to CSVs
    if save_flag:
        df_logits_list_to_logprob_csvs(
            df_logits_list,
            out_dir,
            embeddings_list=embeddings_list,
            float_format=float_format,
        )

    if extract_embeddings:
        return df_logits_list, embeddings_list
    else:
        return df_logits_list


def calc_pos_perplexity(df):
    cols = list("ACDEFGHIKLMNPQRSTVWY")
    t = torch.tensor(df[cols].values)
    probs = F.softmax(t, dim=1)
    perplexities = torch.pow(2, -(probs * torch.log2(probs)).sum(dim=1))

    return perplexities.numpy()


def sample_new_sequences_CDR_HL(
    df,
    t=0.20,
    imgt_regions=["CDR1", "CDR2", "CDR3"],
    exclude_heavy=False,
    exclude_light=False,
    return_mutation_df=False,
    limit_expected_variation=True,
    verbose=False,
):
    """Samples new sequences only varying at H/L CDRs"""

    def _sample_cdr_seq(df, imgt_regions, t=0.20):
        """DF to sampled seq"""

        # CDR1+2+3 mask
        region_mask = get_imgt_mask(df, imgt_regions)

        # Probabilities after scaling with temp
        probs = get_temp_probs(df, t=t)
        probs_cdr = probs[region_mask]

        # Sampled tokens and sequence
        sampled_tokens = torch.multinomial(probs_cdr, 1).squeeze(-1)
        sampled_seq = np.array([amino_list[i] for i in sampled_tokens])

        return sampled_seq

    # Prepare to sample new H + L sequences
    df_H, df_L = get_dfs_HL(df)

    # Get H, sampling only for (CDR1, 2, 3)
    H_sampled = get_df_seq(df_H)

    regions = [region for region in imgt_regions if "L" not in region]
    if len(regions) > 0 and not exclude_heavy:
        region_mask = get_imgt_mask(df_H, regions)
        H_sampled[region_mask] = _sample_cdr_seq(df_H, regions, t=t)

    # Get L, sampling only for (CDR1, 2, 3)
    L_sampled = get_df_seq(df_L)

    regions = [region for region in imgt_regions if "H" not in region]
    if len(regions) > 0 and not exclude_light:
        region_mask = get_imgt_mask(df_L, regions)
        L_sampled[region_mask] = _sample_cdr_seq(df_L, regions, t=t)

    # Use for later
    sampled_seq = np.concatenate([H_sampled, L_sampled])
    region_mask = get_imgt_mask(df, imgt_regions)

    # Mismatches vs predicted (CDR only)
    pred_seq = get_df_seq_pred(df)
    mismatch_idxs_pred_cdr = np.where(
        (sampled_seq[region_mask] != pred_seq[region_mask])
    )[0]

    # Mismatches vs original (all)
    orig_seq = get_df_seq(df)
    mismatch_idxs_orig = np.where((sampled_seq != orig_seq))[0]

    # Limit mutations (backmutate) to as many expected from temperature sampling
    if limit_expected_variation:
        backmutate = len(mismatch_idxs_orig) - len(mismatch_idxs_pred_cdr)

        if backmutate >= 1:
            backmutate_idxs = np.random.choice(
                mismatch_idxs_orig, size=backmutate, replace=False
            )
            sampled_seq[backmutate_idxs] = orig_seq[backmutate_idxs]
            H_sampled = sampled_seq[: len(df_H)]
            L_sampled = sampled_seq[-len(df_L) :]

    # Variables for calculating mismatches
    sampled_seq = np.concatenate([H_sampled, L_sampled])
    orig_seq = get_df_seq(df)

    # DataFrame with sampled mutations
    if return_mutation_df:
        mut_list = np.where(sampled_seq != orig_seq)[0]
        df_mut = df.loc[
            mut_list, ["pdb_res", "top_res", "pdb_posins", "pdb_chain"]
        ].copy()
        df_mut.insert(1, "aa_sampled", sampled_seq[mut_list])

        return H_sampled, L_sampled, df_mut

    return H_sampled, L_sampled, df


def pdb_posins_to_pos(pdb_posins):
    # Convert pos+insertion code to numerical only
    return pdb_posins.astype(str).str.extract(r"(\d+)")[0].astype(int).values


def get_imgt_mask(df, imgt_regions=["CDR1", "CDR2", "CDR3"]):
    """Returns e.g. CDR1+2+3 mask"""

    positions = pdb_posins_to_pos(df["pdb_posins"])
    region_pos_list = list()

    for region in imgt_regions:
        if str(region) not in IMGT_dict.keys():
            region_pos_list.extend(region)
        else:
            region_pos_list.extend(list(IMGT_dict[region]))

    region_mask = pd.Series(positions).isin(region_pos_list).values

    return region_mask


def get_df_logits(df):
    cols = list("ACDEFGHIKLMNPQRSTVWY")
    logits = torch.tensor(df[cols].values)

    return logits


def get_temp_probs(df, t=0.20):
    """Gets temperature scaled probabilities for sampling"""

    logits = get_df_logits(df)
    temp_logits = logits / np.max([t, 0.001])  # Lower-bound to prevent div by 0
    temp_probs = F.softmax(temp_logits, dim=1)

    return temp_probs


def get_dfs_HL(df):
    """Split df into heavy and light chains"""
    Hchain, Lchain = df["pdb_chain"].unique()[:2] # Assume heavy, light
    return df[df["pdb_chain"] == Hchain], df[df["pdb_chain"] == Lchain]


def get_df_seq(df):
    """Get PDB sequence"""
    return df["pdb_res"].values


def get_df_seq_pred(df):
    """Get PDB sequence"""
    return df["top_res"].values


def get_df_seqs_HL(df):
    """Get heavy and light chain sequences"""
    df_H, df_L = get_dfs_HL(df)
    return get_df_seq(df_H), get_df_seq(df_L)


def sample_from_df_logits(
    df_logits,
    sample_n=1,
    sampling_temp=0.20,
    regions_to_mutate=["CDR1", "CDR2", "CDR3"],
    exclude_heavy=False,
    exclude_light=False,
    limit_expected_variation=False,
    verbose=False,
    seed=42,
):
    # Get original H/L sequence
    H_orig, L_orig = get_df_seqs_HL(df_logits)

    # Only sampling from heavy, light chains
    df_logits_HL = df_logits.iloc[:len(H_orig) + len(L_orig), :]
    df_logits_HL.name = df_logits.name

    # Stats
    seq = "".join(H_orig) + "".join(L_orig)
    _, global_score = get_sequence_sampled_global_score(
        seq, df_logits_HL, regions_to_mutate
    )

    # Save to FASTA dict
    fasta_dict = OrderedDict()
    _id = f"{df_logits_HL.name}"
    desc = f", score={global_score:.4f}, global_score={global_score:.4f}, regions={regions_to_mutate}, model_name=AntiFold, seed={seed}"
    seq = "".join(H_orig) + "/" + "".join(L_orig)
    fasta_dict[_id] = SeqIO.SeqRecord(Seq(seq), id=_id, name="", description=desc)

    if verbose:
        log.info(f"{_id}: {desc}")

    if not isinstance(sampling_temp, list):
        sampling_temp = [sampling_temp]

    for t in sampling_temp:
        # Sample sequences n times
        for n in range(sample_n):
            # Get mutated H/L sequence
            H_mut, L_mut, df_mut = sample_new_sequences_CDR_HL(
                df_logits_HL,  # DataFrame with residue probabilities
                t=t,  # Sampling temperature
                imgt_regions=regions_to_mutate,  # Region to sample
                exclude_heavy=exclude_heavy,  # Allow mutations in heavy chain
                exclude_light=exclude_light,  # Allow mutation in light chain
                limit_expected_variation=limit_expected_variation,  # Only mutate as many positions are expected from temperature
                verbose=verbose,
            )

            # Original sequence
            seq_orig = "".join(H_orig) + "".join(L_orig)

            # Sequence recovery and mismatches
            correct_matches = (H_orig == H_mut).sum() + (L_orig == L_mut).sum()
            seq_recovery = correct_matches / len(seq_orig)
            n_mut = (H_orig != H_mut).sum() + (L_orig != L_mut).sum()

            seq_mut = "".join(H_mut) + "".join(L_mut)
            score_sampled, global_score = get_sequence_sampled_global_score(
                seq_mut, df_logits_HL, regions_to_mutate
            )

            # Save to FASTA dict
            _id = f"{df_logits_HL.name}__{n+1}"
            desc = f"T={t:.2f}, sample={n+1}, score={score_sampled:.4f}, global_score={global_score:.4f}, seq_recovery={seq_recovery:.4f}, mutations={n_mut}"
            seq_mut = "".join(H_mut) + "/" + "".join(L_mut)
            fasta_dict[_id] = SeqIO.SeqRecord(
                Seq(seq_mut), id="", name="", description=desc
            )

            if verbose:
                log.info(f"{_id}: {desc}")

    return fasta_dict


def write_fasta_to_dir(fasta_dict, df_logits, out_dir, verbose=True):
    """Write fasta to output folder"""

    os.makedirs(out_dir, exist_ok=True)
    outfile = f"{out_dir}/{df_logits.name}.fasta"
    if verbose:
        log.info(f"Saving to {outfile}")

    with open(outfile, "w") as out_handle:
        SeqIO.write(fasta_dict.values(), out_handle, "fasta")


def visualize_mutations(orig, mut, chain=""):
    """Visualize mutations between two sequences"""

    # Convert to numpy array
    # (whether string, list, Bio.Seq.Seq or np.array)
    orig = np.array(list(orig))
    mut = np.array(list(mut))
    mismatches = "".join(["X" if match else "_" for match in (orig != mut)])

    # Print
    print(f"Mutations ({(orig != mut).sum()}):\t{mismatches}")
    print(f"Original {chain}:\t\t{''.join(orig)}")
    print(f"Mutated {chain}:\t\t{''.join(mut)}\n")


def df_logits_to_probs(df_logits):
    """Convert logits to probabilities"""

    # Calculate probabilities
    amino_list = list("ACDEFGHIKLMNPQRSTVWY")
    t = torch.tensor(df_logits[amino_list].values)
    probs = F.softmax(t, dim=1)

    # Insert into copied dataframe
    df_probs = df_logits.copy()
    df_probs[amino_list] = probs

    return df_probs


def df_logits_to_logprobs(df_logits):
    """Convert logits to probabilities"""

    # Calculate probabilities
    amino_list = list("ACDEFGHIKLMNPQRSTVWY")
    t = torch.tensor(df_logits[amino_list].values)
    probs = F.log_softmax(t, dim=1)

    # Insert into copied dataframe
    df_probs = df_logits.copy()
    df_probs[amino_list] = probs

    return df_probs


def sequence_to_onehot(sequence):
    amino_list = list("ACDEFGHIKLMNPQRSTVWY")
    one_hot = np.zeros((len(sequence), len(amino_list)), dtype=int)
    for i, res in enumerate(sequence):
        one_hot[i, amino_list.index(res)] = 1
    return one_hot


def get_sequence_sampled_global_score(seq, df_logits, regions_to_mutate=False):
    """
    Get average log probability of sampled / all amino acids
    """

    def _scores(S, log_probs, mask):
        criterion = torch.nn.NLLLoss(reduction="none")
        loss = criterion(
            log_probs.contiguous().view(-1, log_probs.size(-1)), S.contiguous().view(-1)
        ).view(S.size())
        scores = torch.sum(loss * mask, dim=-1) / torch.sum(mask, dim=-1)
        return scores

    # One-hot to indices
    onehot = sequence_to_onehot(seq)
    S = torch.argmax(torch.tensor(onehot, dtype=torch.float), dim=-1)

    # Logits to log probs
    logits = torch.tensor(df_logits[amino_list].values)
    log_probs = F.log_softmax(logits, dim=-1)

    # clip probs
    # log_probs = torch.clamp(log_probs, min=-100, max=100)

    # Calculate log odds scores
    mask = torch.ones_like(S, dtype=torch.bool)
    score_global = _scores(S, log_probs, mask)

    if regions_to_mutate:
        region_mask = get_imgt_mask(df_logits, regions_to_mutate)
        mask = torch.tensor(region_mask)
        score_sampled = _scores(S, log_probs, mask)
    else:
        score_sampled = score_global

    return score_sampled.item(), score_global.item()
