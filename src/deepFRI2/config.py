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
        'sequence': 'efficient-capybara-656',
        'kernel': 'iconic-flower-654',
        'fusion': 'gentle-firefly-712',
    },
    'CC': {
        'sequence': 'warm-dawn-692',
        'kernel': 'vital-night-708',
        'fusion': 'hardy-glitter-710',
    },
    'BP': {
        'sequence': 'morning-glade-695',
        'kernel': 'rich-water-709',
        'fusion': 'clean-glade-711',
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
