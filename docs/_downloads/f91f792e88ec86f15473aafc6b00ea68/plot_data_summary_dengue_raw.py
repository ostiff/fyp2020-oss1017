"""
Dataset summary: Dengue
=======================

Report generated using ``dataprep``.
"""

from dataprep.eda import create_report
from pkgname.utils.data_loader import load_dengue
from pkgname.utils.print_utils import suppress_stdout, suppress_stderr

features = ["dsource", "age", "gender", "weight", "bleeding", "plt",
            "shock", "haematocrit_percent", "bleeding_gum", "abdominal_pain",
            "ascites", "bleeding_mucosal", "bleeding_skin", "body_temperature"]

with suppress_stdout() and suppress_stderr():
    df = load_dengue(usecols=features)
    report = create_report(df, title="Dengue dataset report")

report
