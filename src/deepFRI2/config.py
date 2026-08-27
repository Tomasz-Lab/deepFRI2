"""Immutable deepFRI2 model configuration.

This module contains ONLY the constants that define the released deepFRI2 model:
run names, ontologies, embedding/kernel dimensions and the GO ontology version.
Its content is hashed to derive the model version (see ``version.py``), so keep it
free of runtime/environment-dependent values (paths, tunable inference options,
device selection, ...). Those belong in the caller / CLI, not here.
"""

# Model's params

MODEL_NAMES = {
    'MF': {
        'sequence': 'comfy-silence-824',
        'structure': 'good-glitter-814',
        'fusion': 'dainty-deluge-829',
    },
    'CC': {
        'sequence': 'valiant-waterfall-833',
        'structure': 'daily-firefly-834',
        'fusion': 'divine-donkey-835',
    },
    'BP': {
        'sequence': 'autumn-shape-836',
        'structure': 'comic-cloud-837',
        'fusion': 'leafy-silence-838',
    }
}

DIST_TYPE = 'CA'
MAX_SEQ_LEN = 1020
ONTOLOGIES = list(MODEL_NAMES.keys())

# Sequence analyzer's params

EMBED_MODEL_NAME = 'esm2_t33_650M_UR50D'
ESM_DIM = 1280

# Structural prober's params

SIGMA_DIST = 10
M_DIAG = 60
M_ANTI = 60
NUM_DIAG = 30
NUM_ANTI = 30

# TF32 backend flag for the structural prober's convolutions which keeps 
# convolution outputs reproducible across GPUs / cuDNN versions and against CPU.
CUDNN_ALLOW_TF32 = False

# GO ontology version (selects params/go_<GO_VERSION>.obo for hierarchy propagation)
GO_VERSION = '20250722'
