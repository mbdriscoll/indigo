import os
import subprocess
import pytest

has_spmm_py = os.path.exists("./examples/spmm.py")

@pytest.mark.skipif( not has_spmm_py,
    reason="requires running pytest from slo root directory")
def test_spmm():
    subprocess.run(
        "python ./examples/spmm.py",
        shell=True, check=True)


has_scan_data = os.path.exists("./scan.h5")
has_pics_py   = os.path.exists("./examples/pics.py")

@pytest.mark.skipif( not (has_pics_py and has_scan_data),
    reason="requires running pytest from slo root directory")
def test_pics():
    subprocess.run(
        "python ./examples/pics.py --crop TIME:1,COIL:4 -i 2 --debug 10 scan.h5",
        shell=True, check=True)
