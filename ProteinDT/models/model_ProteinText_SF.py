import torch.nn as nn
import torch
from .model_Protein_SF import ProteinModel_SF
from .model_Text_SF import TextModel_SF

class ProteinTextModel_SF(nn.Module):
    def __init__(self, protein_model_SF: ProteinModel_SF, text_model_SF: TextModel_SF):
        super().__init__()
        self.protein_model_SF = protein_model_SF
        self.text_model_SF = text_model_SF

    def forward(self, protein_sequence_input_ids, protein_sequence_attention_mask, 
                structure_text_input_ids, structure_text_attention_mask, 
                functional_text_input_ids, functional_text_attention_mask):
        protein_structure_repr, protein_functional_repr = self.protein_model_SF(protein_sequence_input_ids, protein_sequence_attention_mask)
        text_structure_repr, text_functional_repr = self.text_model_SF(structure_text_input_ids, structure_text_attention_mask, functional_text_input_ids, functional_text_attention_mask)
        return text_structure_repr, text_functional_repr, protein_structure_repr, protein_functional_repr

class ProteinTextModel_SF_T_Only(nn.Module):
    def __init__(self, protein_model, text_model, protein2latent_model, text2structure_latent_model, text2functional_latent_model):
        super().__init__()
        self.protein_model = protein_model
        self.text_model = text_model
        self.protein2latent_model = protein2latent_model
        self.text2structure_latent_model = text2structure_latent_model
        self.text2functional_latent_model = text2functional_latent_model
    
    def forward(self, protein_sequence_input_ids, protein_sequence_attention_mask, structure_text_input_ids, structure_text_attention_mask, functional_text_input_ids, functional_text_attention_mask):
        protein_output = self.protein_model(protein_sequence_input_ids, protein_sequence_attention_mask)
        protein_repr = protein_output["pooler_output"]
        try:
            protein_repr = self.protein2latent_model(protein_repr)
        except:
            # This is a sklearn model
            protein_repr = torch.tensor(self.protein2latent_model.fit_transform(protein_repr.cpu())).to(protein_repr.device)

        structure_output = self.text_model(structure_text_input_ids, structure_text_attention_mask)
        structure_repr = structure_output["pooler_output"]
        try:
            structure_repr = self.text2structure_latent_model(structure_repr)
        except:
            # This is a sklearn model
            structure_repr = torch.tensor(self.text2structure_latent_model.fit_transform(structure_repr.cpu())).to(structure_repr.device)

        functional_output = self.text_model(functional_text_input_ids, functional_text_attention_mask)
        functional_repr = functional_output["pooler_output"]
        try:
            functional_repr = self.text2functional_latent_model(functional_repr)
        except:
            # This is a sklearn model
            functional_repr = torch.tensor(self.text2functional_latent_model.fit_transform(functional_repr.cpu())).to(functional_repr.device)

        return protein_repr, structure_repr, functional_repr