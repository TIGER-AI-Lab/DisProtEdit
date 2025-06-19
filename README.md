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

---

## 📰 News

- **2025 Jun 18**: Paper available on Arxiv.
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
