import os
import random
import numpy as np
import argparse
from tqdm import tqdm
import time
import signal
import sys
try:
    import wandb
    logger = wandb
except ImportError:
    logger = None

import torch
from torch import nn
import torch.optim as optim

from transformers import BertTokenizer, T5Tokenizer
from torch.utils.data import DataLoader


from ProteinDT.datasets import RepresentationPairWithRawDataDataset
# from ProteinDT.models import GaussianSDEDecoderModel, ColdDiffusionDecoder, LatentDiffusionDecoder, MultinomialDiffusion, LSTMDecoder, T5Decoder
from ProteinDT.models import MultinomialDiffusion, T5Decoder
import warnings

# Suppress all FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def signal_handler(sig, frame):
    print('Training interrupted. Saving checkpoint...')
    model_file = f"model_epoch_{e}.pth"
    saved_file_path = os.path.join(args.output_model_dir, f"decoder_distribution_{model_file}")
    torch.save(decoder_distribution_model.state_dict(), saved_file_path)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def compute_accuracy(original_sequences, generated_sequences, num_examples=5):
    """Compute sequence reconstruction accuracy at sequence and character level"""
    seq_correct = 0
    seq_total = 0
    char_correct = 0
    char_total = 0
    reconstructed_examples = []
    
    for orig, gen in zip(original_sequences[:num_examples], generated_sequences[:num_examples]):  # Only log first num_examples examples
        orig = orig.replace("<pad>", "").strip()
        gen = gen.replace("<pad>", "").strip()
        
        # Sequence level accuracy
        seq_match = orig == gen
        seq_correct += int(seq_match)
        seq_total += 1
        
        # Character level accuracy
        min_len = min(len(orig), len(gen))
        char_matches = 0
        for i in range(min_len):
            if orig[i] == gen[i]:
                char_matches += 1
                char_correct += 1
        char_total += len(orig)  # Use original sequence length as reference
        char_match = char_matches/len(orig) * 100 if len(orig) > 0 else 0
        
        reconstructed_examples.append({
            "original": orig,
            "generated": gen,
            "seq_match": seq_match,
            "char_match": char_match
        })
        
    seq_accuracy = seq_correct/seq_total * 100
    char_accuracy = char_correct/char_total * 100 if char_total > 0 else 0
    
    return seq_accuracy, char_accuracy, reconstructed_examples

def save_model(save_best):
    # create output dir if not exists
    if not os.path.exists(args.output_model_dir):
        os.makedirs(args.output_model_dir)

    if save_best:
        global optimal_loss
        print("save model with loss: {:.5f}".format(optimal_loss))
        model_file = "model.pth"
        
        saved_file_path = os.path.join(args.output_model_dir, "decoder_distribution_{}".format(model_file))
        torch.save(decoder_distribution_model.state_dict(), saved_file_path)

    else:
        model_file = "model_final.pth"
        
        saved_file_path = os.path.join(args.output_model_dir, "decoder_distribution_{}".format(model_file))
        torch.save(decoder_distribution_model.state_dict(), saved_file_path)
    return

def train_AMP(dataloader):
    scaler = torch.cuda.amp.GradScaler()
    decoder_distribution_model.train()
    if args.verbose:
        L = tqdm(dataloader)
    else:
        L = dataloader

    start_time = time.time()
    accum_SDE_loss, accum_decoding_loss = 0, 0
    reconstruction_accuracies = []
    
    for batch_idx, batch in enumerate(L):
        protein_sequence = batch["protein_sequence"]
        batch_size = len(protein_sequence)
        protein_sequence_encode = protein_decoder_tokenizer(protein_sequence, truncation=True, max_length=args.protein_max_sequence_len, padding='max_length', return_tensors='pt')
        protein_sequence_input_ids = protein_sequence_encode.input_ids.squeeze(1).to(device)
        protein_sequence_attention_mask = protein_sequence_encode.attention_mask.squeeze(1).to(device)
        
        protein_repr = batch["protein_repr"].to(device)

        with torch.cuda.amp.autocast():
            SDE_loss, decoding_loss = decoder_distribution_model(protein_seq_input_ids=protein_sequence_input_ids, protein_seq_attention_mask=protein_sequence_attention_mask, condition=protein_repr)
            loss = args.alpha_1 * SDE_loss + args.alpha_2 * decoding_loss

        if args.verbose and batch_idx % 100 == 0:
            if torch.is_tensor(decoding_loss):
                print("SDE Loss: {:.5f}\tDecoding Loss: {:.5f}".format(SDE_loss.item(), decoding_loss.item()))
            else:
                print("SDE Loss: {:.5f}\tDecoding Loss: {:.5f}".format(SDE_loss.item(), decoding_loss))

            # Do reconstruction for logging
            with torch.no_grad():
                generated_ids = decoder_distribution_model.inference(
                    condition=protein_repr,
                    max_seq_len=args.protein_max_sequence_len,
                    protein_seq_attention_mask=protein_sequence_attention_mask
                )
                generated_sequences = protein_decoder_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                original_sequences = protein_sequence
                #print(len(generated_sequences)) # 438982
                #print(len(original_sequences)) # 438982
                char_accuracy, seq_accuracy, examples = compute_accuracy(original_sequences, generated_sequences, num_examples=batch_size)
                print(f"Character Accuracy: {char_accuracy:.2f}%, Sequence Accuracy: {seq_accuracy:.2f}%")
                reconstruction_accuracies.append(char_accuracy)  # Keep using char accuracy for consistency
                
                if logger:
                    logger.log({
                        "reconstruction_examples": examples,
                        "batch_char_accuracy": char_accuracy,
                        "batch_seq_accuracy": seq_accuracy
                    })

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        accum_SDE_loss += SDE_loss.item()
        if torch.is_tensor(decoding_loss):
            accum_decoding_loss += decoding_loss.item()

    accum_SDE_loss /= len(L)
    accum_decoding_loss /= len(L)
    avg_seq_reconstruction_accuracy = np.mean(reconstruction_accuracies) if reconstruction_accuracies else 0
    
    global optimal_loss
    temp_loss = args.alpha_1 * accum_SDE_loss + args.alpha_2 * decoding_loss
    if temp_loss < optimal_loss:
        optimal_loss = temp_loss
        save_model(save_best=True)
    print("SDE Loss: {:.5f}\tDecoding Loss: {:.5f}\tReconstruction Accuracy: {:.2f}%\tTime: {:.5f}".format(
        accum_SDE_loss, accum_decoding_loss, avg_seq_reconstruction_accuracy, time.time() - start_time))
    
    # Log metrics to wandb
    if logger:
        logger.log({
            "SDE_loss": accum_SDE_loss,
            "decoding_loss": accum_decoding_loss,
            "total_loss": temp_loss,
            "epoch_time": time.time() - start_time,
            "reconstruction_accuracy": avg_seq_reconstruction_accuracy
        })
    return {
        "avg_SDE_loss": accum_SDE_loss / len(L),
        "avg_decoding_loss": accum_decoding_loss / len(L), 
        "avg_total_loss": temp_loss / len(L),
        "reconstruction_accuracy": avg_seq_reconstruction_accuracy
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--condition_dim", type=int, default=256)
    parser.add_argument("--protein_backbone_model", type=str, default="ProtBert", choices=["ProtBert", "ProtBert_BFD"])
    parser.add_argument("--protein_max_sequence_len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decay", type=float, default=0)

    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.set_defaults(verbose=False)
    
    parser.add_argument("--use_AMP", dest="use_AMP", action="store_true")
    parser.add_argument("--no_AMP", dest="use_AMP", action="store_false")
    parser.set_defaults(use_AMP=True)

    parser.add_argument("--decoder_distribution", type=str, default="T5Decoder", choices=["T5Decoder", "MultinomialDiffusion"])
    parser.add_argument("--pretrained_folder", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--output_model_dir", type=str, default=None)
    parser.add_argument("--target_subfolder", type=str, default="pairwise_all")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint to resume training from")

    # for GaussianSDE & diffusion
    parser.add_argument("--beta_min", type=float, default=0.1)
    parser.add_argument("--beta_max", type=float, default=30)
    parser.add_argument("--num_diffusion_timesteps", type=int, default=1000)
    parser.add_argument("--SDE_type", type=str, default="VP")
    parser.add_argument("--score_network_type", type=str, default="Toy")
    parser.add_argument("--alpha_1", type=float, default=1)
    parser.add_argument("--alpha_2", type=float, default=0)
    parser.add_argument("--prob_unconditional", type=float, default=0)

    args = parser.parse_args()
    print("arguments", args)
    assert args.pretrained_folder is not None
    assert args.output_model_dir is not None

    if logger:
        logger.init(
            project="protein-decoder",
            config=vars(args),
            name=args.wandb_name
        )

    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    
    if args.decoder_distribution in ["T5Decoder"]:
        protein_decoder_tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False, chache_dir="../data/temp_pretrained_t5_base")
        print(protein_decoder_tokenizer.get_vocab())
    else:
        protein_decoder_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False, chache_dir="../data/temp_pretrained_ProtBert")

    if args.decoder_distribution == "MultinomialDiffusion":
        mask_id = 4
        decoder_distribution_model = MultinomialDiffusion(
            hidden_dim=args.hidden_dim,
            condition_dim=args.condition_dim, mask_id=mask_id,
            beta_min=args.beta_min, beta_max=args.beta_max, num_diffusion_timesteps=args.num_diffusion_timesteps,
            num_classes=protein_decoder_tokenizer.vocab_size, score_network_type=args.score_network_type)

    elif args.decoder_distribution == "T5Decoder":
        decoder_distribution_model = T5Decoder(
            hidden_dim=args.condition_dim,
            tokenizer=protein_decoder_tokenizer,
            T5_model=args.score_network_type)

    if torch.cuda.device_count() > 1:
        # parallel models
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(decoder_distribution_model)
        neo_batch_size = args.batch_size * torch.cuda.device_count()
        print("batch size from {} to {}".format(args.batch_size, neo_batch_size))
        args.batch_size = neo_batch_size
    decoder_distribution_model.to(device)

    # Load checkpoint if provided
    if args.checkpoint_path is not None:
        print(f"Loading checkpoint from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path)
        decoder_distribution_model.load_state_dict(checkpoint)

    model_param_group = [
        {"params": decoder_distribution_model.parameters(), "lr": args.lr},
    ]
    optimizer = optim.Adam(model_param_group, weight_decay=args.decay)
    optimal_loss = 1e10

    step_02_folder = os.path.join(args.pretrained_folder, "step_02_pairwise_representation")
    step_02_folder = os.path.join(step_02_folder, args.target_subfolder)
    dataset = RepresentationPairWithRawDataDataset(step_02_folder, prob_unconditional=args.prob_unconditional, repr_filename="pairwise_representation_P_T", empty_filename="empty_sequence")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    for e in range(1, args.epochs+1):
        print("Epoch {}".format(e))
        if args.use_AMP:
            epoch_dict = train_AMP(dataloader)
        else:
            raise NotImplementedError
        if logger: 
            logger.log(epoch_dict)
