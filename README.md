# UCSF-PDGM Tumor Classification

Frozen ViT + linear classifier pipeline for volumetric brain tumor classification.

## Usage (run on Colab / GPU environment)
```python
!git clone https://github.com/<your-username>/ucsf-pdgm-classifier.git
%cd ucsf-pdgm-classifier
!pip install -r requirements.txt -q

from downstream import downstream

checkpoint = downstream(
    model_id="google/vit-base-patch16-224",
    dataset_id="chehablab/UCSF_PDGM",
    allowed_labels=["Glioblastoma, IDH-wildtype", "Astrocytoma, IDH-mutant", ""],
)
```