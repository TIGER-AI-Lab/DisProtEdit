import torch.nn as nn
import torch

class ProteinModel_SF(nn.Module):
    def __init__(self, protein_model, protein2structure_latent_model, protein2functional_latent_model):
        super().__init__()
        self.protein_model = protein_model
        self.protein2structure_latent_model = protein2structure_latent_model
        self.protein2functional_latent_model = protein2functional_latent_model

    def forward(self, protein_input_ids, protein_attention_mask):
        structure_output = self.protein_model(protein_input_ids, protein_attention_mask)
        structure_repr = structure_output["pooler_output"]
        try:
            structure_repr = self.protein2structure_latent_model(structure_repr)
        except:
            # This is a sklearn model
            structure_repr = torch.tensor(self.protein2structure_latent_model.fit_transform(structure_repr.cpu())).to(structure_repr.device)

        functional_output = self.protein_model(protein_input_ids, protein_attention_mask)
        functional_repr = functional_output["pooler_output"]
        try:
            functional_repr = self.protein2functional_latent_model(functional_repr)
        except:
            # This is a sklearn model
            functional_repr = torch.tensor(self.protein2functional_latent_model.fit_transform(functional_repr.cpu())).to(functional_repr.device)

        return structure_repr, functional_repr