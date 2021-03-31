import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')

DENGUE_DATASET_VERSION = '20210313-v0.0.8'
DENGUE_PATH = os.path.join(DATA_DIR, 'OUCRU', DENGUE_DATASET_VERSION, 'combined', 'combined_tidy.csv')
PATHOLOGY_PATH = os.path.join(DATA_DIR, 'daily-profile.csv')