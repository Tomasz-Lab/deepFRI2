"""deepFRI2 version derived from the immutable model configuration.

The version is a SHA-256 hash of the *values* of the public constants in ``config.py``
(canonicalised as sorted JSON), not of the file's raw bytes. This makes the version
insensitive to whitespace, comments and formatting (e.g. running black): it changes
only when an actual configuration value changes.
"""

import hashlib
import json

import config


def config_payload():
    """Return the canonical JSON string of config's public constants (sorted, stable)."""
    values = {name: getattr(config, name) for name in dir(config) if name.isupper()}
    return json.dumps(values, sort_keys=True, default=str)


def config_version(short=12):
    """Return the deepFRI2 config version (hex digest; truncated to ``short`` chars if set)."""
    digest = hashlib.sha256(config_payload().encode("utf-8")).hexdigest()
    return digest[:short] if short else digest


__version__ = config_version()
