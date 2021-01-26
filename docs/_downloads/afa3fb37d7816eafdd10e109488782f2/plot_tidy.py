"""
TidyWidget
===========================
"""
# Import
import numpy as np
import pandas as pd

# DataBlend library
from pkgname.utils.widgets import TidyWidget

# ------------------------
# Constants
# ------------------------
# Transformed data
data = [
    {'id': '32dx-001', 'date': '2020/12/05', 'column': 'bt', 'result': 37.2, 'unit': 'celsius'},
    {'id': '32dx-001', 'date': '2020/12/05', 'column': 'age', 'result': 32, 'unit': 'year'},
    {'id': '32dx-001', 'date': '2020/12/05', 'column': 'gender', 'result': 'Male'},
    {'id': '32dx-001', 'date': '2020/12/05', 'column': 'pregnant', 'result': False},
    {'id': '32dx-001', 'date': '2020/12/05', 'column': 'pcr_dengue_serotype', 'result': 'DENV-1'},
    {'id': '32dx-001', 'date': '2020/12/05', 'column': 'pcr_dengue_serotype', 'result': 'DENV-2'},
    {'id': '32dx-002', 'date': '2020/12/05', 'column': 'bt', 'result': 38.2},
    {'id': '32dx-002', 'date': '2020/12/05', 'column': 'bt', 'result': 39.7},
    {'id': '32dx-002', 'date': '2020/12/05', 'column': 'age', 'result': np.nan},
    {'id': '32dx-002', 'date': '2020/12/05', 'column': 'gender', 'result': 'Female'},
    {'id': '32dx-002', 'date': '2020/12/05', 'column': 'pregnant', 'result': True},
    {'id': '32dx-002', 'date': '2020/12/05', 'column': 'pregnant', 'result': False},
    {'id': '32dx-002', 'date': '2020/12/05', 'column': 'pcr_dengue_serotype', 'result': 'DENV-3'},
    {'id': '32dx-002', 'date': '2020/12/05', 'column': 'pcr_dengue_serotype', 'result': 'DENV-4'},
    {'id': '32dx-002', 'date': '2020/12/05', 'column': 'wbc', 'result': '15'},
    {'id': '32dx-002', 'date': '2020/12/05', 'column': 'wbc', 'result': '18'},
    {'id': '32dx-003', 'date': '2020/12/05', 'column': 'vomiting', 'result': 'False'},
    {'id': '32dx-003', 'date': '2020/12/05', 'column': 'vomiting', 'result': False},
    {'id': '32dx-004', 'date': '2020/12/05', 'column': 'vomiting', 'result': 'False'},
    {'id': '32dx-004', 'date': '2020/12/05', 'column': 'vomiting', 'result': False},
    {'id': '32dx-004', 'date': '2020/12/05', 'column': 'vomiting', 'result': 'True'},
    {'id': '32dx-004', 'date': '2020/12/05', 'column': 'vomiting', 'result': True},
]

# Parameters
index = ['id', 'date', 'column']
value = 'result'

# --------------------
# Main
# --------------------
# Create data
data = pd.DataFrame(data)

# Create widget
widget = TidyWidget(index=index, value=value)

# Transform (keep all)
transform, duplicated = \
    widget.transform(data, report_duplicated=True)

# Transform (keep first)
transform_first = \
    widget.transform(data, keep='first')

# Show
print("\nStacked:")
print(data)
print("\nDuplicated:")
print(duplicated)
print("\nTidy (all):")
print(transform)
print("\nTidy (first):")
print(transform_first)
print("\nDtypes:")
print(transform.dtypes)