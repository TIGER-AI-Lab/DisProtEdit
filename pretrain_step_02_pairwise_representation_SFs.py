import os
import random
import numpy as np
import argparse
from tqdm import tqdm
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset

from ProteinDT.models import ProteinTextModel_SF_T_Only
from ProteinDT.datasets import SwissProtCLAP_SFDataset
import warnings

#import pdb

# Suppress all FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

@torch.no_grad()
def extract_AMP(dataloader):
    if args.verbose:
        L = tqdm(dataloader)
    else:
        L = dataloader
    
    protein_repr_list, structure_repr_list, functional_repr_list, protein_seq_list, text_S_seq_list, text_F_seq_list = [], [], [], [], [], []
    text_seq_list = []
    for batch_idx, batch in enumerate(L):
        protein_seq = batch["protein_sequence"]
        text_seq = batch["text_sequence"]
        text_S_seq = batch["structure_text"]
        text_F_seq = batch["functional_text"]
        protein_sequence_input_ids = batch["protein_sequence_input_ids"].to(device)
        protein_sequence_attention_mask = batch["protein_sequence_attention_mask"].to(device)
        text_sequence_input_ids = batch["text_sequence_input_ids"].to(device)
        text_sequence_attention_mask = batch["text_sequence_attention_mask"].to(device)
        structure_text_input_ids = batch["structure_text_input_ids"].to(device)
        structure_text_attention_mask = batch["structure_text_attention_mask"].to(device)
        functional_text_input_ids = batch["functional_text_input_ids"].to(device)
        functional_text_attention_mask = batch["functional_text_attention_mask"].to(device)

        
        with torch.cuda.amp.autocast():
            protein_repr, structure_repr, functional_repr = model(protein_sequence_input_ids, protein_sequence_attention_mask, structure_text_input_ids, structure_text_attention_mask, functional_text_input_ids, functional_text_attention_mask)
        protein_repr_list.append(protein_repr.detach().cpu().numpy())
        structure_repr_list.append(structure_repr.detach().cpu().numpy())
        functional_repr_list.append(functional_repr.detach().cpu().numpy())
        protein_seq_list.extend(protein_seq)
        text_S_seq_list.extend(text_S_seq)
        text_F_seq_list.extend(text_F_seq)
        text_seq_list.extend(text_seq)

    protein_repr_array = np.concatenate(protein_repr_list)
    structure_repr_array= np.concatenate(structure_repr_list)
    functional_repr_array= np.concatenate(functional_repr_list)
    return protein_repr_array, structure_repr_array, functional_repr_array, protein_seq_list, text_S_seq_list, text_F_seq_list, text_seq_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--SSL_emb_dim", type=int, default=256)
    parser.add_argument("--protein_backbone_model", type=str, default="ProtBERT_BFD", choices=["ProtBERT", "ProtBERT_BFD"])
    parser.add_argument("--protein_max_sequence_len", type=int, default=512)
    parser.add_argument("--text_max_sequence_len", type=int, default=512)
    
    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.set_defaults(verbose=False)
    
    parser.add_argument("--use_AMP", dest="use_AMP", action="store_true")
    parser.add_argument("--no_AMP", dest="use_AMP", action="store_false")
    parser.set_defaults(use_AMP=True)

    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--sample_portion", type=float, default=None)

    parser.add_argument("--pretrained_folder", type=str, default=None)
    parser.add_argument("--target_subfolder", type=str, default=None)
    parser.add_argument("--ds_llm", type=str, default="gemini")
    parser.add_argument("--ds_name", type=str, default="500k", choices=["random_10k", "500k"])

    args = parser.parse_args()
    print("arguments", args)

    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ##### Load pretrained protein model
    if args.protein_backbone_model == "ProtBERT":
        protein_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False, chache_dir="../data/temp_pretrained_ProtBert")
        protein_model = BertModel.from_pretrained("Rostlab/prot_bert", cache_dir="../data/temp_pretrained_ProtBert")
    elif args.protein_backbone_model == "ProtBERT_BFD":
        protein_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False, chache_dir="../data/temp_pretrained_ProtBert_BFD")
        protein_model = BertModel.from_pretrained("Rostlab/prot_bert_bfd", cache_dir="../data/temp_pretrained_ProtBert_BFD")
    protein_dim = 1024
    input_model_path = os.path.join(args.pretrained_folder, "protein_model.pth")
    print("Loading protein model from {}...".format(input_model_path))
    state_dict = torch.load(input_model_path, map_location='cpu', weights_only=True)
    protein_model.load_state_dict(state_dict)

    ##### Load pretrained text model
    text_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir="../data/temp_pretrained_SciBert")
    text_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir="../data/temp_pretrained_SciBert")
    text_dim  = 768
    input_model_path = os.path.join(args.pretrained_folder, "text_model.pth")
    print("Loading text model from {}...".format(input_model_path))
    state_dict = torch.load(input_model_path, map_location='cpu', weights_only=True)
    text_model.load_state_dict(state_dict)

    ##### Load pretrained protein2latent model
    protein2latent_model = nn.Linear(protein_dim, args.SSL_emb_dim)
    input_model_path = os.path.join(args.pretrained_folder, "protein2latent_model.pth")
    print("Loading protein2latent model from {}...".format(input_model_path))
    state_dict = torch.load(input_model_path, map_location='cpu', weights_only=True)
    protein2latent_model.load_state_dict(state_dict)

    ##### Load pretrained text2latent model
    text2latent_S_model = nn.Linear(text_dim, args.SSL_emb_dim//2)
    input_model_path = os.path.join(args.pretrained_folder, "text2latent_S_model.pth")
    print("Loading text2latent model from {}...".format(input_model_path))
    state_dict = torch.load(input_model_path, map_location='cpu', weights_only=True)
    text2latent_S_model.load_state_dict(state_dict)

    text2latent_F_model = nn.Linear(text_dim, args.SSL_emb_dim//2)
    input_model_path = os.path.join(args.pretrained_folder, "text2latent_F_model.pth")
    print("Loading text2latent model from {}...".format(input_model_path))
    state_dict = torch.load(input_model_path, map_location='cpu', weights_only=True)
    text2latent_F_model.load_state_dict(state_dict)

    model = ProteinTextModel_SF_T_Only(protein_model, text_model, protein2latent_model, text2latent_S_model, text2latent_F_model)
    model.eval()
    model.to(device)

    #hf_ds = load_dataset(f"vinesmsuic/SwissProtCLAP_{args.ds_name}_{args.ds_llm}")
    hf_ds = load_dataset("TIGER-Lab/SwissProtDis_500k")
    dataset = SwissProtCLAP_SFDataset(
        hf_ds=hf_ds,
        protein_tokenizer=protein_tokenizer,
        text_tokenizer=text_tokenizer,
        protein_max_sequence_len=args.protein_max_sequence_len,
        text_max_sequence_len=args.text_max_sequence_len
    )
    if args.num_samples is not None:
        sampled_dataset = torch.utils.data.Subset(dataset, np.random.choice(len(dataset), size=args.num_samples, replace=False))
    elif args.sample_portion is not None:
        sampled_dataset = torch.utils.data.Subset(dataset, np.random.choice(len(dataset), size=int(args.sample_portion * len(dataset)), replace=False))
    else:
        sampled_dataset = dataset
    dataloader = DataLoader(sampled_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if args.use_AMP:
        protein_repr_array, structure_repr_array, functional_repr_array, protein_seq_list, text_S_seq_list, text_F_seq_list, text_seq_list = extract_AMP(dataloader)
    #concatenate the structure_repr_array and functional_repr_array as text_repr_array
    text_repr_array = np.concatenate([structure_repr_array, functional_repr_array], axis=1)
    print("text_repr_array shape", text_repr_array.shape)

    assert args.pretrained_folder is not None
    output_folder = os.path.join(args.pretrained_folder, "step_02_pairwise_representation")
    if args.target_subfolder is not None:
        output_folder = os.path.join(output_folder, args.target_subfolder)

    os.makedirs(output_folder, exist_ok=True)

    saved_file_path = os.path.join(output_folder, "pairwise_representation_P_T")
    np.savez(saved_file_path, protein_repr_array=protein_repr_array, description_repr_array=text_repr_array)

    saved_file_path = os.path.join(output_folder, "pairwise_representation_S_F")
    np.savez(saved_file_path, protein_repr_array=structure_repr_array, description_repr_array=functional_repr_array)

    protein_sequence_file = os.path.join(output_folder, "protein_sequence.txt")
    f = open(protein_sequence_file, 'w')
    for protein_seq in protein_seq_list:
        print(protein_seq, file=f)

    text_sequence_file = os.path.join(output_folder, "text_S_sequence.txt")
    f = open(text_sequence_file, 'w')
    for text_seq in text_S_seq_list:
        print(text_seq, file=f)

    text_sequence_file = os.path.join(output_folder, "text_F_sequence.txt")
    f = open(text_sequence_file, 'w')
    for text_seq in text_F_seq_list:
        print(text_seq, file=f)

    text_sequence_file = os.path.join(output_folder, "text_sequence.txt")
    f = open(text_sequence_file, 'w')
    for text_seq in text_seq_list:
        print(text_seq, file=f)