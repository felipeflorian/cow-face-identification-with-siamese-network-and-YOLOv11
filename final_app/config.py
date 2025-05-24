# config.py
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY         = "change-this"
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

    UPLOAD_FOLDER      = os.path.join(BASE_DIR, "app", "static", "uploads")
    EMBEDDING_THRESHOLD= 0.9

    YOLO_WEIGHTS       = os.path.join(BASE_DIR, "models", "best_cow_recognizer.pt")
    EMBEDDER_WEIGHTS   = os.path.join(BASE_DIR, "models", "cow_embedder_resnet18.pt")
    GALLERY_PTH        = os.path.join(BASE_DIR, "models", "gallery_db.pth")
