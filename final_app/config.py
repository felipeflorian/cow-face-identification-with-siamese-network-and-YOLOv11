from pathlib import Path
import os
BASE_DIR = os.path.abspath(os.path.dirname(__file__))


ROOT = Path(__file__).resolve().parent
class Config:
    SECRET_KEY = "change-this"
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB
    UPLOAD_FOLDER = ROOT / "app" / "static" / "uploads"
    EMBEDDING_THRESHOLD = 0.6
    #RECOGNIZER_WEIGHTS = os.path.join(BASE_DIR, "models", "best_cow_recognizer.pt")
