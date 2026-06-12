import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "PSSP.keras")
MAX_LEN = 128

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INT = {aa: i + 1 for i, aa in enumerate(AA_ORDER)}
INT_TO_AA = {i + 1: aa for i, aa in enumerate(AA_ORDER)}

STRUCTURE_LABELS = ["H", "E", "C", "_"]
