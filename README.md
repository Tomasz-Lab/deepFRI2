# deepFRI2

deepFRI2 is an upgraded version of the well-established [deepFRI](https://www.nature.com/articles/s41467-021-23303-9) (Deep Functional Residue Identification) framework for predicting protein function using [Gene Ontology](https://geneontology.org/) (GO) terms and [Enzyme Commission](https://enzyme.expasy.org/) (EC) numbers.

Like its predecessor, deepFRI2 operates in two complementary modes: sequence-based and sequence–structure-based. This dual approach enables robust functional inference in metagenomic settings — where protein structures are often unavailable — as well as high-precision annotation when structural information is known.

For training, deepFRI2 leverages [FRIData](https://github.com/Tomasz-Lab/FRIdata), a scalable and efficient library for generating large, non-redundant protein datasets.

While maintaining similar input and output formats, the model architecture has been completely redesigned to incorporate recent advances in the field, particularly the use of protein language models as powerful representations of protein sequences.

**The model is currently under development.** Once finalized, the full architecture will be open-sourced, along with all components required for retraining and fine-tuning.
