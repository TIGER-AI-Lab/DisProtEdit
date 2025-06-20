# 🧬 DisProtEdit

[![arXiv](https://img.shields.io/badge/arXiv-2506.14853-b31b1b.svg)](https://arxiv.org/abs/2506.14853)

[**🌐 Homepage**](https://tiger-ai-lab.github.io/DisProtEdit/)  | [**📖 arXiv**](https://arxiv.org/abs/2506.14853) 

[![license](https://img.shields.io/github/license/TIGER-AI-Lab/DisProtEdit.svg)](https://github.com/TIGER-AI-Lab/DisProtEdit/blob/main/LICENSE)
[![GitHub](https://img.shields.io/github/stars/TIGER-AI-Lab/DisProtEdit?style=social)](https://github.com/TIGER-AI-Lab/DisProtEdit)

This repo contains the codebase for our paper:  
**DisProtEdit: Exploring Disentangled Representations for Multi-Attribute Protein Editing**

**📍 ICML 2025 Workshops (GenBio, FM4LS)**

---

## 📌 Introduction

DisProtEdit is a protein editing framework that disentangles structural and functional properties using dual-channel natural language supervision. It learns modular latent representations aligned with protein sequences through a combination of alignment, uniformity, and angular MMD losses. Editing is performed via text modification, enabling interpretable and controllable edits to protein structure or function.

![](https://tiger-ai-lab.github.io/DisProtEdit/static/images/method.png)


---

## 📰 News

- **2025 Jun 20**: Released SwissProtDis Dataset, Editing benchmark, also the full training code.
- **2025 Jun 18**: Paper available on Arxiv.
- **2025 Jun 17**: Website created!  
- **2025 Jun 11**: DisProtEdit accepted to ICMLW GenBio.
- **2025 Jun 10**: DisProtEdit accepted to ICMLW FM4LS.  

---

## 📦 SwissProtDis Dataset

We introduce **SwissProtDis**, a large-scale multimodal dataset containing:
- ~540,000 protein sequences
- Automatically decomposed structural and functional text descriptions from UniProt using GPT-4o

👉 [https://huggingface.co/datasets/TIGER-Lab/SwissProtDis_500k](https://huggingface.co/datasets/TIGER-Lab/SwissProtDis_500k)

---

## Environment Setup

```
conda create -n disprot python=3.10
conda activate disprot
pip install -r requirements.txt
```

## Training

Training multimodal embeddings
```shell
./1_SFs.sh 
./2_empty_SFs.sh 
./2_sample_SFs.sh
```

Alternatively, you can run:
```shell
export OUTPUT_DIR="output/"
export PRETRAINED_DIR="output/SFs_AU_b24_gpt4o_500k_DisAngle10"

CUDA_VISIBLE_DEVICES=0,1,2,3\
        python3 pretrain_step_01_SFs.py \
        --protein_lr=1e-5 --protein_lr_scale=1 \
        --text_lr=1e-5 --text_lr_scale=1 --CL_loss="EBM_NCE"\
        --protein_backbone_model=ProtBERT_BFD --wandb_name="SFs_AU05_b24_gpt4o_500k"\
        --epochs=10 --batch_size=24 --num_workers=0 --verbose \
        --output_model_dir="$OUTPUT_DIR" --CL=0.0 --D=0.0 --U=0.5 --A=1.0 --dis_angle --ds_llm="gpt4o" --ds_name="500k"

python3 pretrain_step_02_empty_sequence_SFs.py \
--protein_backbone_model=ProtBERT_BFD \
--batch_size=16 --num_workers=4 \
--pretrained_folder="$PRETRAINED_DIR" \
--target_subfolder="pairwise_all"

python3 pretrain_step_02_pairwise_representation_SFs.py \
--protein_backbone_model=ProtBERT_BFD \
--batch_size=16 --num_workers=4 \
--pretrained_folder="$PRETRAINED_DIR" \
--target_subfolder="pairwise_all" \
--ds_llm="gpt4o"

```

Training decoder for editing task
```shell
./4_SFs.sh
```

Alternatively, you can run:
```shell
export PRETRAINED_DIR="output/SFs_AU_b24_gpt4o_500k_DisAngle10"
CUDA_VISIBLE_DEVICES=0\
        python pretrain_step_04_decoder_SFs.py \
        --batch_size=8 --lr=1e-4 --epochs=10 \
        --decoder_distribution=T5Decoder \
        --score_network_type=T5Base --wandb_name="SFs_AU_b24_gpt4o_500k_DisAngle10"\
        --hidden_dim=16  --verbose \
        --pretrained_folder="$PRETRAINED_DIR" \
        --output_model_dir="$PRETRAINED_DIR"/step_04_T5 \
        --target_subfolder="pairwise_all"
```

## 🧪 Editing Benchmark

Please see [_datasets_and_checkpoints](https://github.com/TIGER-AI-Lab/DisProtEdit/blob/main/_datasets_and_checkpoints).

## Downstream Tasks

### Editing

Multi-Attribute Protein Editing
```shell
./5_medit.sh
```

### TAPE

Protein Properties Prediction
```shell
./5_TAPE.sh
```

The code is built upon [TAPE in ProteinDT](https://github.com/chao1224/ProteinDT/blob/main/examples/downstream_TAPE.py).

---


## 📖 Citation

```bibtex
@misc{ku2025disproteditexploringdisentangledrepresentations,
      title={DisProtEdit: Exploring Disentangled Representations for Multi-Attribute Protein Editing}, 
      author={Max Ku and Sun Sun and Hongyu Guo and Wenhu Chen},
      year={2025},
      booktitle={ICML Workshop on Generative AI and Biology},
      eprint={2506.14853},
      archivePrefix={arXiv},
      primaryClass={q-bio.QM},
      url={https://arxiv.org/abs/2506.14853}, 
}
```

## 💞 Acknowledgements
This code is heavily built upon [ProteinDT](https://github.com/chao1224/ProteinDT). we thank all the contributors for open-sourcing.



