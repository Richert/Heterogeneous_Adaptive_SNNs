r"""
Shared output/data locations for the PRL_2026 manuscript pipelines
===================================================================

Replaces the ``/home/rgast/data/...`` literals that used to be hard-coded in every
sweep and figure script.

``mpmf_simulations`` lives on the shared lab drive (``SHARED_MPMF``) so that every
machine reads and writes the same results.  The other directories live under the
data root (default ``~/data``).  Overrides, in order of precedence::

    HASNN_MPMF=/some/dir python ...      # mpmf_simulations only
    HASNN_DATA=/scratch/$USER/data ...   # data root (mpmf_simulations = <root>/mpmf_simulations)

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

#: root of all generated data; override with ``HASNN_DATA``
ROOT = os.environ.get("HASNN_DATA") or os.path.join(os.path.expanduser("~"), "data")

#: shared lab-drive location of ``mpmf_simulations`` (used unless overridden)
SHARED_MPMF = "/mnt/kennedy_labdata/richard_turbulence/data/mpmf_simulations"

MPMF = (os.environ.get("HASNN_MPMF")
        or (os.path.join(os.environ["HASNN_DATA"], "mpmf_simulations")
            if os.environ.get("HASNN_DATA") else SHARED_MPMF))
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
