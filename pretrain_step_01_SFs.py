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
import math
from transformers import AutoModel, AutoTokenizer
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset

from ProteinDT.models import ProteinTextModel_SF_T_Only
from ProteinDT.datasets import SwissProtCLAP_SFDataset

from losses import lalign, lunif, l_mmd

try:
    import wandb
    logger = wandb
except ImportError:
    logger = None



def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr


def do_CL(X, Y, args):
    if args.normalize:
        X = F.normalize(X, dim=-1)
        Y = F.normalize(Y, dim=-1)

    if args.CL_loss == 'EBM_NCE':
        # EBM_NCE: The predictions are based on the similarity scores of positive and negative pairs, 
        # and the accuracy reflects how well the model distinguishes between these pairs.

        criterion = nn.BCEWithLogitsLoss()
        neg_Y = torch.cat([Y[cycle_index(len(Y), i + 1)] for i in range(args.CL_neg_samples)], dim=0)
        neg_X = X.repeat((args.CL_neg_samples, 1))

        # similarity score for positive pairs and negative pairs
        pred_pos = torch.sum(X * Y, dim=1) / args.T
        pred_neg = torch.sum(neg_X * neg_Y, dim=1) / args.T

        loss_pos = criterion(pred_pos, torch.ones(len(pred_pos)).to(pred_pos.device))
        loss_neg = criterion(pred_neg, torch.zeros(len(pred_neg)).to(pred_neg.device))
        CL_loss = (loss_pos + args.CL_neg_samples * loss_neg) / (1 + args.CL_neg_samples)

        # The accuracy is computed by counting how many positive predictions are greater than 0 
        # (indicating they are classified as positive) and 
        # how many negative predictions are less than 0 (indicating they are classified as negative).
        CL_acc = (torch.sum(pred_pos > 0).float() + torch.sum(pred_neg < 0).float()) / \
                 (len(pred_pos) + len(pred_neg))
        CL_acc = CL_acc.detach().cpu().item()

    elif args.CL_loss == 'InfoNCE':
        # InfoNCE: The predictions are class labels derived from a softmax output, 
        # and the accuracy reflects how well the model classifies each sample against all others.
        criterion = nn.CrossEntropyLoss()
        B = X.size()[0]
        logits = torch.mm(X, Y.transpose(1, 0))  # B*B
        logits = torch.div(logits, args.T)
        labels = torch.arange(B).long().to(logits.device)  # B*1

        # The predicted class labels are obtained by taking the index of the maximum logit for each sample.
        # The accuracy is computed by comparing the predicted labels with the true labels and 
        # summing the correct predictions, then dividing by the total number of samples B.
        CL_loss = criterion(logits, labels)
        pred = logits.argmax(dim=1, keepdim=False)
        CL_acc = pred.eq(labels).sum().detach().cpu().item() * 1. / B

    else:
        raise Exception

    return CL_loss, CL_acc

def do_align(X, Y, alpha=2):
    align_loss = lalign(X, Y, alpha)
    return align_loss

def do_uniform(X, t=2):
    uniform_loss = lunif(X, t)
    return uniform_loss

def do_disentanglement(Z_1, Z_2):
    batch_size_1, latent_dim_1 = Z_1.size(0), Z_1.size(1)
    batch_size_2, latent_dim_2 = Z_2.size(0), Z_2.size(1)
    x_1 = torch.randn(batch_size_1, latent_dim_1).to(device)
    x_2 = torch.randn(batch_size_2, latent_dim_2).to(device)
    disentanglement_loss = l_mmd(x_1, Z_1) + l_mmd(x_2, Z_2)
    return disentanglement_loss

def do_disentanglement_angle(Z_1, Z_2, device='cuda', sample_angle=True):
    if sample_angle: # Randomly sample phi in the range [0, pi/2] to ensure r1^2 + r2^2 = 1
        phi = torch.rand(1).item() * (math.pi / 2)  # phi ~ Uniform(0, pi/2)
        r1 = math.cos(phi)  # r1 = cos(phi), ensures r1^2 + r2^2 = 1 
        r2 = math.sin(phi)  # r2 = sin(phi), since cos^2(theta) + sin^2(theta) = 1
    else: # r1^2 + r2^2 = 1
        r1 = math.sqrt(0.5)
        r2 = math.sqrt(0.5)

    batch_size_1, latent_dim_1 = Z_1.size(0), Z_1.size(1)
    batch_size_2, latent_dim_2 = Z_2.size(0), Z_2.size(1)
    x_1 = torch.randn(batch_size_1, latent_dim_1).to(device)  # x_1 ~ N(0, 1)
    x_2 = torch.randn(batch_size_2, latent_dim_2).to(device)  # x_2 ~ N(0, 1)
    
    # Normalize vectors to unit norm
    x_1 = x_1 / torch.norm(x_1, dim=1, keepdim=True, p=2)  # Normalize x_1 with L2 norm
    x_2 = x_2 / torch.norm(x_2, dim=1, keepdim=True, p=2)  # Normalize x_2 with L2 norm

    # Compute disentanglement loss (e.g., MMD or other metric)
    disentanglement_loss = l_mmd(r1 * x_1, Z_1) + l_mmd(r2 * x_2, Z_2)
    
    return disentanglement_loss

def save_model(save_best):
    # create output dir if not exists
    if not os.path.exists(args.output_model_dir):
        os.makedirs(args.output_model_dir)
    
    if args.output_model_dir is None:
        return
    
    if save_best:
        global optimal_loss
        print("save model with loss: {:.5f}".format(optimal_loss))
        model_file = "model.pth"
        
        saved_file_path = os.path.join(args.output_model_dir, "text_{}".format(model_file))
        torch.save(text_model.state_dict(), saved_file_path)
        
        saved_file_path = os.path.join(args.output_model_dir, "protein_{}".format(model_file))
        torch.save(protein_model.state_dict(), saved_file_path)
        
        saved_file_path = os.path.join(args.output_model_dir, "text2latent_S_{}".format(model_file))
        torch.save(text2latent_S_model.state_dict(), saved_file_path)

        saved_file_path = os.path.join(args.output_model_dir, "text2latent_F_{}".format(model_file))
        torch.save(text2latent_F_model.state_dict(), saved_file_path)
        
        saved_file_path = os.path.join(args.output_model_dir, "protein2latent_{}".format(model_file))
        torch.save(protein2latent_model.state_dict(), saved_file_path)

    else:
        model_file = "model_final.pth"

        saved_file_path = os.path.join(args.output_model_dir, "text_{}".format(model_file))
        torch.save(text_model.state_dict(), saved_file_path)
        
        saved_file_path = os.path.join(args.output_model_dir, "protein_{}".format(model_file))
        torch.save(protein_model.state_dict(), saved_file_path)
        
        saved_file_path = os.path.join(args.output_model_dir, "text2latent_S_{}".format(model_file))
        torch.save(text2latent_S_model.state_dict(), saved_file_path)
        
        saved_file_path = os.path.join(args.output_model_dir, "text2latent_F_{}".format(model_file))
        torch.save(text2latent_F_model.state_dict(), saved_file_path)
        
        saved_file_path = os.path.join(args.output_model_dir, "protein2latent_{}".format(model_file))
        torch.save(protein2latent_model.state_dict(), saved_file_path)

    return

def train_AMP(dataloader):
    scaler = torch.amp.GradScaler("cuda")

    if args.verbose:
        L = tqdm(dataloader)
    else:
        L = dataloader
    
    start_time = time.time()
    accum_loss, accum_acc = 0, 0
    for batch_idx, batch in enumerate(L):
        protein_sequence_input_ids = batch["protein_sequence_input_ids"].to(device)
        protein_sequence_attention_mask = batch["protein_sequence_attention_mask"].to(device)
        text_sequence_input_ids = batch["text_sequence_input_ids"].to(device)
        text_sequence_attention_mask = batch["text_sequence_attention_mask"].to(device)
        structure_text_input_ids = batch["structure_text_input_ids"].to(device)
        structure_text_attention_mask = batch["structure_text_attention_mask"].to(device)
        functional_text_input_ids = batch["functional_text_input_ids"].to(device)
        functional_text_attention_mask = batch["functional_text_attention_mask"].to(device)

        
        with torch.amp.autocast("cuda"):
            protein_repr, structure_repr, functional_repr = model(protein_sequence_input_ids, protein_sequence_attention_mask, 
                                                   structure_text_input_ids, structure_text_attention_mask,
                                                   functional_text_input_ids, functional_text_attention_mask)
            
            text_repr = torch.cat((structure_repr, functional_repr), dim=1)
            loss_01, acc_01 = do_CL(text_repr, protein_repr, args)
            loss_02, acc_02 = do_CL(protein_repr, text_repr, args)

            cl_loss = args.CL * (loss_01 + loss_02) / 2
            acc = (acc_01 + acc_02) / 3
            align_loss = args.A * do_align(text_repr, protein_repr)
            text_uniform_loss = args.U * (do_uniform(structure_repr) + do_uniform(functional_repr))
            prot_uniform_loss = args.U * do_uniform(protein_repr)
            prot_loss = prot_uniform_loss
            text_loss = text_uniform_loss
            if args.dis_angle:
                disentanglement_loss = args.D * do_disentanglement_angle(structure_repr, functional_repr)
            else:
                disentanglement_loss = args.D * do_disentanglement(structure_repr, functional_repr)
            total_loss = cl_loss + align_loss + prot_loss + text_loss + disentanglement_loss

        optimizer.zero_grad()
        scaler.scale(total_loss).backward()  # Retain the graph for the next backward pass
        scaler.step(optimizer)
        scaler.update()

        accum_loss = accum_loss + cl_loss.item()
        accum_acc = accum_acc + acc
        if args.verbose and batch_idx % 100 == 0:
            print("", "cl loss:", cl_loss.item(), "| acc:", acc)

            if logger:
                logger.log({"CL Loss": cl_loss.item(), "CL Acc": acc})
                logger.log({"Alignment Loss": align_loss.item()})
                logger.log({"Prot Uniform Loss": prot_uniform_loss.item(), "Text Uniform Loss": text_uniform_loss.item()})
                logger.log({"Prot Total Loss": prot_loss.item(), "Text Total Loss": text_loss.item()})
                logger.log({"Disentanglement Loss": disentanglement_loss.item()})

    accum_loss = accum_loss/len(L)
    accum_acc = accum_acc/len(L)
    global optimal_loss
    temp_loss = accum_loss
    if temp_loss < optimal_loss:
        optimal_loss = temp_loss
        save_model(save_best=True)
    print("CL Loss: {:.5f}\tCL Acc: {:.5f}Time: {:.5f}".format(accum_loss, accum_acc, time.time() - start_time))
    #return average metrics at the end of the epoch
    return {"avg_CL Loss": accum_loss / len(L), "avg_CL Acc": accum_acc / len(L)}



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--SSL_emb_dim", type=int, default=256)
    parser.add_argument("--protein_backbone_model", type=str, default="ProtBERT_BFD", choices=["ProtBERT", "ProtBERT_BFD"])
    parser.add_argument("--protein_max_sequence_len", type=int, default=512)
    parser.add_argument("--text_max_sequence_len", type=int, default=512)
    parser.add_argument("--protein_lr", type=float, default=1e-5)
    parser.add_argument("--protein_lr_scale", type=float, default=1e-1)
    parser.add_argument("--text_lr", type=float, default=1e-5)
    parser.add_argument("--text_lr_scale", type=float, default=1e-1)
    parser.add_argument("--CL_neg_samples", type=int, default=1)
    parser.add_argument("--CL_loss", type=str, default="EBM_NCE")
    parser.add_argument("--T", type=float, default=0.1)
    parser.add_argument("--U", type=float, default=0.2)
    parser.add_argument("--A", type=float, default=1.0)
    parser.add_argument("--D", type=float, default=0.1) # 1e-1 or 1e-2
    parser.add_argument("--CL", type=float, default=1.0)
    parser.add_argument("--dis_angle", dest="dis_angle", action="store_true")
    parser.add_argument("--decay", type=float, default=0)
    parser.add_argument("--ds_llm", type=str, default="gemini")
    parser.add_argument("--ds_name", type=str, default="500k", choices=["random_10k", "500k"])

    parser.add_argument("--normalize", dest="normalize", action="store_true")
    parser.add_argument("--no_normalize", dest="normalize", action="store_false")
    parser.set_defaults(normalize=False)

    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.set_defaults(verbose=False)
    
    
    parser.add_argument("--use_AMP", dest="use_AMP", action="store_true")
    parser.add_argument("--no_AMP", dest="use_AMP", action="store_false")
    parser.set_defaults(use_AMP=True)

    parser.add_argument("--output_model_dir", type=str, default=None)

    parser.add_argument("--wandb_name", type=str, default="default")

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

    args.output_model_dir = os.path.join(args.output_model_dir, args.wandb_name)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    if args.protein_backbone_model == "ProtBERT":
        protein_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False, chache_dir="../data/temp_pretrained_ProtBert")
        protein_model = BertModel.from_pretrained("Rostlab/prot_bert", cache_dir="../data/temp_pretrained_ProtBert")
    elif args.protein_backbone_model == "ProtBERT_BFD":
        protein_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False, chache_dir="../data/temp_pretrained_ProtBert_BFD")
        protein_model = BertModel.from_pretrained("Rostlab/prot_bert_bfd", cache_dir="../data/temp_pretrained_ProtBert_BFD")
    protein_dim = 1024

    # TODO: check https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py#L1501
    text_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir="../data/temp_pretrained_SciBert")
    text_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir="../data/temp_pretrained_SciBert")
    text_dim  = 768

    protein2latent_model = nn.Linear(protein_dim, args.SSL_emb_dim)
    text2latent_S_model = nn.Linear(text_dim, args.SSL_emb_dim // 2)
    text2latent_F_model = nn.Linear(text_dim, args.SSL_emb_dim // 2)

    model = ProteinTextModel_SF_T_Only(protein_model, text_model, protein2latent_model, text2latent_S_model, text2latent_F_model)

    if torch.cuda.device_count() > 1:
        # parallel models
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        neo_batch_size = args.batch_size * torch.cuda.device_count()
        print("batch size from {} to {}".format(args.batch_size, neo_batch_size))
        args.batch_size = neo_batch_size
    model.to(device)

    model_param_group = [
        {"params": protein_model.parameters(), "lr": args.protein_lr * args.protein_lr_scale},
        {"params": text_model.parameters(), "lr": args.text_lr * args.text_lr_scale},
        {"params": protein2latent_model.parameters(), "lr": args.protein_lr * args.protein_lr_scale},
        {"params": text2latent_S_model.parameters(), "lr": args.text_lr * args.text_lr_scale},
        {"params": text2latent_F_model.parameters(), "lr": args.text_lr * args.text_lr_scale},
    ]

    optimizer = optim.Adam(model_param_group, weight_decay=args.decay)
    optimal_loss = 1e10

    
    #hf_ds = load_dataset("vinesmsuic/SwissProtCLAP_" + str(args.ds_name) + "_" + str(args.ds_llm))
    hf_ds = load_dataset("TIGER-Lab/SwissProtDis_500k")
    dataset = SwissProtCLAP_SFDataset(
        hf_ds=hf_ds,
        protein_tokenizer=protein_tokenizer,
        text_tokenizer=text_tokenizer,
        protein_max_sequence_len=args.protein_max_sequence_len,
        text_max_sequence_len=args.text_max_sequence_len
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    if logger:
        logger.init(project="proteindt_SFs_stage1", name=args.wandb_name)
        logger.config.update({
            "learning_rate": args.protein_lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
        })

    for e in range(1, args.epochs+1):
        print("Epoch {}".format(e))
        if args.use_AMP:
            epoch_dict = train_AMP(dataloader)
        else:
            raise NotImplementedError
        if logger: 
            logger.log(epoch_dict)

