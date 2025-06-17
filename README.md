# 🧬 DisProtEdit: Disentangled Representations for Protein Editing

[![arXiv](https://img.shields.io/badge/arXiv-TBA-b31b1b.svg)](https://arxiv.org/TBA)

[**🌐 Homepage**](https://tiger-ai-lab.github.io/DisProtEdit/)  | [**📖 arXiv**](https://arxiv.org/TBA) | [**🤗 HuggingFace Dataset**](https://huggingface.co/datasets/TBA) | [🧪 Multi-Edit Benchmark](https://huggingface.co/datasets/TBA)

[![license](https://img.shields.io/github/license/TIGER-AI-Lab/DisProtEdit.svg)](https://github.com/TIGER-AI-Lab/DisProtEdit/blob/main/LICENSE)
[![GitHub](https://img.shields.io/github/stars/TIGER-AI-Lab/DisProtEdit?style=social)](https://github.com/TIGER-AI-Lab/DisProtEdit)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FTIGER-AI-Lab%2FDisProtEdit&count_bg=%237A91C0&title_bg=%23555555&icon=github.svg&icon_color=%23E7E7E7&title=visitors&edge_flat=false)](https://hits.seeyoufarm.com)

This repo contains the codebase for our paper:  
**DisProtEdit: Exploring Disentangled Representations for Multi-Attribute Protein Editing**

**📍 ICML 2025 Workshops (GenBio, FM4LS)**

---

## 📌 Introduction

DisProtEdit is a protein editing framework that disentangles structural and functional properties using dual-channel natural language supervision. It learns modular latent representations aligned with protein sequences through a combination of alignment, uniformity, and angular MMD losses. Editing is performed via text modification, enabling interpretable and controllable edits to protein structure or function.

---

## 📰 News

- **2025 Jun 17**: Website created!  
- **2025 Jun 11**: DisProtEdit accepted to ICMLW GenBio.
- **2025 Jun 10**: DisProtEdit accepted to ICMLW FM4LS.  

---

## 📦 SwissProtDis Dataset

We introduce **SwissProtDis**, a large-scale multimodal dataset containing:
- ~540,000 protein sequences
- Automatically decomposed structural and functional text descriptions from UniProt using GPT-4o

👉 [TBA: Hugging Face dataset link]

---

## 🧪 Editing Benchmark

_**Under construction**_

---


## 📖 Citation

```bibtex
@inproceedings{ku2025disprotedit,
  title={DisProtEdit: Exploring Disentangled Representations for Multi-Attribute Protein Editing},
  author={Max Ku and Sun Sun and Hongyu Guo and Wenhu Chen},
  booktitle={ICML Workshop on Generative AI and Biology},
  year={2025}
}
