import torch.nn as nn
import torch

class TextModel_SF(nn.Module):
    def __init__(self, text_model, text2structure_latent_model, text2functional_latent_model):
        super().__init__()
        self.text_model = text_model
        self.text2structure_latent_model = text2structure_latent_model
        self.text2functional_latent_model = text2functional_latent_model

    def forward(self, structure_text_input_ids, structure_text_attention_mask, functional_text_input_ids, functional_text_attention_mask):
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

        return structure_repr, functional_repr