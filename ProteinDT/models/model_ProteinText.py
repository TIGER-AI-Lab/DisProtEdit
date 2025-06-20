import torch.nn as nn
import torch

class ProteinTextModel(nn.Module):
    def __init__(self, protein_model, text_model, protein2latent_model, text2latent_model):
        super().__init__()
        self.protein_model = protein_model
        self.text_model = text_model
        self.protein2latent_model = protein2latent_model
        self.text2latent_model = text2latent_model
        return
    
    def forward(self, protein_sequence_input_ids, protein_sequence_attention_mask, text_sequence_input_ids, text_sequence_attention_mask):
        protein_output = self.protein_model(protein_sequence_input_ids, protein_sequence_attention_mask)
        protein_repr = protein_output["pooler_output"]
        try:
            protein_repr = self.protein2latent_model(protein_repr)
        except:
            # This is a sklearn model
            protein_repr = torch.tensor(self.protein2latent_model.fit_transform(protein_repr.cpu())).to(protein_repr.device)

        description_output = self.text_model(text_sequence_input_ids, text_sequence_attention_mask)
        description_repr = description_output["pooler_output"]
        try:
            description_repr = self.text2latent_model(description_repr)
        except:
            # This is a sklearn model
            description_repr = torch.tensor(self.text2latent_model.fit_transform(description_repr.cpu())).to(description_repr.device)
        return protein_repr, description_repr