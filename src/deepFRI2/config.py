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
        'sequence': 'fast-dream-733',
        'structure': 'eager-wind-756',
        'fusion': 'expert-surf-758',
    },
    'CC': {
        'sequence': 'denim-firefly-739',
        'structure': 'stellar-elevator-757',
        'fusion': 'cerulean-frost-762',
    },
    'BP': {
        'sequence': 'major-dream-736',
        'structure': 'trim-serenity-759',
        'fusion': 'expert-valley-761',
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

# GO ontology version (selects params/go_<GO_VERSION>.obo for hierarchy propagation)

GO_VERSION = '20250722'
