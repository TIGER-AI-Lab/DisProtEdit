import os
import torch
from torch.utils.data import Dataset

class SwissProtCLAP_SFDataset(Dataset):
    def __init__(self, hf_ds, protein_tokenizer, text_tokenizer, protein_max_sequence_len, text_max_sequence_len):
        self.hf_ds = hf_ds['train']
        self.protein_tokenizer = protein_tokenizer
        self.text_tokenizer = text_tokenizer
        self.protein_max_sequence_len = protein_max_sequence_len
        self.text_max_sequence_len = text_max_sequence_len

        print("Parsing SwissProtCLAPDataset... With customized HF Dataset")
        
        # Filter the dataset
        filtered_hf_ds = self.hf_ds.filter(lambda x: len(x['Protein Sequence']) <= self.protein_max_sequence_len and 
                                           x['Protein Sequence'] and x['gt_desc'] and x['structure_info'] and x['functional_info'])
        protein_sequence_list = [" ".join(entry['Protein Sequence']) for entry in filtered_hf_ds]
        text_sequence_list = [entry['gt_desc'] for entry in filtered_hf_ds]
        structure_text_list = [entry['structure_info'] for entry in filtered_hf_ds]
        functional_text_list = [entry['functional_info'] for entry in filtered_hf_ds]

        # Convert to lists
        self.protein_sequence_list = list(protein_sequence_list)
        self.text_sequence_list = list(text_sequence_list)
        self.structure_text_list = list(structure_text_list)
        self.functional_text_list = list(functional_text_list)
        print("num of (protein-sequence, text, structure-text, functional-text) pair: {}".format(len(self.protein_sequence_list)))

        return

    def __getitem__(self, index):
        protein_sequence = self.protein_sequence_list[index]
        text_sequence = self.text_sequence_list[index]
        structure_text = self.structure_text_list[index]
        functional_text = self.functional_text_list[index]

        protein_sequence_encode = self.protein_tokenizer(
            text=protein_sequence,
            truncation=True,
            max_length=self.protein_max_sequence_len,
            padding='max_length',
            return_tensors='pt'
        )
        protein_sequence_input_ids = protein_sequence_encode.input_ids.squeeze()
        protein_sequence_attention_mask = protein_sequence_encode.attention_mask.squeeze()

        text_sequence_encode = self.text_tokenizer(
            text=text_sequence,
            truncation=True,
            max_length=self.text_max_sequence_len,
            padding='max_length',
            return_tensors='pt'
        )
        text_sequence_input_ids = text_sequence_encode.input_ids.squeeze()
        text_sequence_attention_mask = text_sequence_encode.attention_mask.squeeze()

        structure_text_encode = self.text_tokenizer(
            text=structure_text,
            truncation=True,
            max_length=self.text_max_sequence_len,
            padding='max_length',
            return_tensors='pt'
        )
        structure_text_input_ids = structure_text_encode.input_ids.squeeze()
        structure_text_attention_mask = structure_text_encode.attention_mask.squeeze()

        functional_text_encode = self.text_tokenizer(
            text=functional_text,
            truncation=True,
            max_length=self.text_max_sequence_len,
            padding='max_length',
            return_tensors='pt'
        )
        functional_text_input_ids = functional_text_encode.input_ids.squeeze()
        functional_text_attention_mask = functional_text_encode.attention_mask.squeeze()
        
        batch = {
            "protein_sequence": protein_sequence,
            "protein_sequence_input_ids": protein_sequence_input_ids,
            "protein_sequence_attention_mask": protein_sequence_attention_mask,
            "text_sequence": text_sequence,
            "text_sequence_input_ids": text_sequence_input_ids,
            "text_sequence_attention_mask": text_sequence_attention_mask,
            "structure_text": structure_text,
            "structure_text_input_ids": structure_text_input_ids,
            "structure_text_attention_mask": structure_text_attention_mask,
            "functional_text": functional_text,
            "functional_text_input_ids": functional_text_input_ids,
            "functional_text_attention_mask": functional_text_attention_mask,
        }

        return batch
    
    def __len__(self):
        return len(self.protein_sequence_list)


if __name__ == "__main__":
    from transformers import AutoModel, AutoTokenizer, BertModel, BertTokenizer
    from torch.utils.data import DataLoader
    from datasets import load_dataset

    hf_ds = load_dataset("vinesmsuic/SwissProtCLAP_random_10k_gemini")

    protein_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False, chache_dir="../../data/temp_pretrained_ProtBert")

    text_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir="../../data/temp_pretrained_SciBert")

    dataset = SwissProtCLAP_SFDataset(
        hf_ds=hf_ds,
        protein_tokenizer=protein_tokenizer,
        text_tokenizer=text_tokenizer,
        protein_max_sequence_len=512,
        text_max_sequence_len=512
    )
    print("len of dataset", len(dataset))
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    for batch in dataloader:
        protein_sequence_list = batch["protein_sequence"]
        text_sequence_list = batch["text_sequence"] 
        structure_text_list = batch["structure_text"]
        functional_text_list = batch["functional_text"]
        
        for protein_sequence, text_sequence, structure_text, functional_text in zip(protein_sequence_list, text_sequence_list, structure_text_list, functional_text_list):
            print(protein_sequence.replace(" ", ""))
            print(text_sequence)
            print(structure_text)
            print(functional_text)
        break
