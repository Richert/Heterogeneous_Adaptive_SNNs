r"""
Download the SCN single-cell PER2::LUC data of Abel et al. (2016), PNAS 113:4512-4517
=====================================================================================

"Functional network inference of the suprachiasmatic nucleus", data repository
https://github.com/JohnAbel/scn-resynchronization-data-2016 (hourly bioluminescence of individual
SCN neurons in 5 slices; tetrodotoxin (TTX) applied and washed out to desynchronize and
resynchronize the network).

The repository archive is downloaded and unpacked to data_paths.MPMF/circadian_data/raw/
(shared data drive, not the git repository).

    python download_scn_data.py [--force]
"""
import io
import os
import sys
import urllib.request
import zipfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "shared"))
import data_paths as dp

URL = "https://github.com/JohnAbel/scn-resynchronization-data-2016/archive/refs/heads/master.zip"
RAW = dp.ensure(dp.mpmf("circadian_data", "raw"))


def download(force=False):
    target = os.path.join(RAW, "scn-resynchronization-data-2016-master")
    if os.path.isdir(target) and not force:
        print(f"[download] already present: {target}")
        return target
    print(f"[download] {URL}")
    with urllib.request.urlopen(URL, timeout=120) as r:
        data = r.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(RAW)
    print(f"[download] {len(data) / 1e6:.1f} MB unpacked to {target}")
    return target


if __name__ == "__main__":
    root = download("--force" in sys.argv)
    for d, _, files in sorted(os.walk(root)):
        for f in sorted(files):
            p = os.path.join(d, f)
            print(f"  {os.path.relpath(p, root):70s} {os.path.getsize(p) / 1e3:9.1f} kB")
