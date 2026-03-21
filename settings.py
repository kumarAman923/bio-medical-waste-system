from pathlib import Path
import sys

file_path = Path(__file__).resolve()
root_path = file_path.parent
if root_path not in sys.path:
    sys.path.append(str(root_path))
ROOT = root_path.relative_to(Path.cwd())

# ML Model config
MODEL_DIR = ROOT / 'weights'
DETECTION_MODEL = MODEL_DIR / 'best.pt'

# Webcam
WEBCAM_PATH = 0


# -----------------------------
# Biomedical Waste Categories
# -----------------------------

# Infectious waste
INFECTIOUS = [
    "syringe",
    "needle",
    "bandage"
]

# Protective equipment
PROTECTIVE = [
    "mask",
    "gloves"
]

# Medical plastic waste
MEDICAL_PLASTIC = [
    "medicine_bottle"
]