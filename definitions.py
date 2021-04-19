import os

# Root
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Data
DATA_DIR = os.path.join(ROOT_DIR, 'data')
DENGUE_DATASET_VERSION = '20210401-v0.0.9'
DENGUE_PATH = os.path.join(DATA_DIR, 'OUCRU', DENGUE_DATASET_VERSION, 'combined', 'combined_tidy.csv')
PATHOLOGY_PATH = os.path.join(DATA_DIR, 'daily-profile.csv')

# Logs
LOG_PATH = '/vol/bitbucket/oss1017/logs'
TEMPLATES_PATH = os.path.join(ROOT_DIR, 'pkgname', 'utils', 'templates')