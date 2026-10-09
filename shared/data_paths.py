r"""
Shared output/data locations for the PRL_2026 manuscript pipelines
===================================================================

Replaces the ``/home/rgast/data/...`` literals that used to be hard-coded in every
sweep and figure script.

``mpmf_simulations`` lives on the shared lab drive so that every machine reads and writes
the same results.  The drive is mounted at different points on different machines; the
first existing entry of ``SHARED_MPMF_CANDIDATES`` is used (add a line for a new machine).
The other directories live under the data root (default ``~/data``).  Resolution order
for ``MPMF``::

    HASNN_MPMF=/some/dir python ...      # 1. explicit override (mpmf_simulations only)
    HASNN_DATA=/scratch/$USER/data ...   # 2. data root -> <root>/mpmf_simulations
                                         # 3. first existing SHARED_MPMF_CANDIDATES entry
                                         # 4. ~/data/mpmf_simulations (with a warning)

Directories
-----------
``MPMF``          large sweeps + most manuscript figures (``mpmf_simulations``)
``KMO_ADAPTIVE``  adaptive-Kuramoto ramps and coherence sweeps (``kmo_adaptive``)
``QIF_PLASTICITY``  QIF/STDP simulation output (``qif_plasticity``)

Helpers
-------
``mpmf("stem")`` etc. join a stem onto the directory and return a ``str`` (the
scripts pass these straight to ``np.savez`` / ``fig.savefig``).  ``ensure(d)``
creates a directory if needed and returns it.
"""

import os
import warnings

#: root of all generated data; override with ``HASNN_DATA``
ROOT = os.environ.get("HASNN_DATA") or os.path.join(os.path.expanduser("~"), "data")

#: mount points of the shared lab drive's ``mpmf_simulations`` on the different machines
SHARED_MPMF_CANDIDATES = [
    "/mnt/kennedy_labdata/richard_turbulence/data/mpmf_simulations",   # workstation
    "/media/storage/DATA/richard_turbulence/data/mpmf_simulations",    # compute server
]


def _resolve_mpmf():
    if os.environ.get("HASNN_MPMF"):
        return os.environ["HASNN_MPMF"]
    if os.environ.get("HASNN_DATA"):
        return os.path.join(os.environ["HASNN_DATA"], "mpmf_simulations")
    for cand in SHARED_MPMF_CANDIDATES:
        if os.path.isdir(cand):
            return cand
    fallback = os.path.join(ROOT, "mpmf_simulations")
    warnings.warn(f"shared mpmf_simulations drive not found at any of {SHARED_MPMF_CANDIDATES}; "
                  f"using {fallback} (set HASNN_MPMF to choose explicitly)", stacklevel=2)
    return fallback


MPMF = _resolve_mpmf()
KMO_ADAPTIVE = os.path.join(ROOT, "kmo_adaptive")
QIF_PLASTICITY = os.path.join(ROOT, "qif_plasticity")


def ensure(directory):
    """Create ``directory`` if it does not exist and return it."""
    os.makedirs(directory, exist_ok=True)
    return directory


def mpmf(*parts):
    """Path inside the ``mpmf_simulations`` output directory."""
    return os.path.join(MPMF, *parts)


def kmo_adaptive(*parts):
    """Path inside the ``kmo_adaptive`` output directory."""
    return os.path.join(KMO_ADAPTIVE, *parts)


def qif_plasticity(*parts):
    """Path inside the ``qif_plasticity`` output directory."""
    return os.path.join(QIF_PLASTICITY, *parts)
