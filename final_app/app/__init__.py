from flask import Flask
from pathlib import Path
from .recognizer import CowRecognizer
from config import Config

def create_app():
    app = Flask(__name__, instance_relative_config=False)
    app.config.from_object("config.Config")

    app.config.from_object(Config)


    # # 1) Inicializa BD y recognizer
    # db_path = Path(app.instance_path) / "embeddings.db"
    # db_path.parent.mkdir(parents=True, exist_ok=True)
    # app.embedding_db = EmbeddingDB(str(db_path))

    # recognizer = CowRecognizer(weights=app.config["RECOGNIZER_WEIGHTS"])
    
    # # 2) Asegura que la galería de embeddings esté poblada
    # gallery_root = app.config["GALLERY_FOLDER"]  # pon aquí tu carpeta de sub-dirs por vaca
    # ensure_gallery_embeddings(app.embedding_db, recognizer, gallery_root)



    # Asegúrate de que exista la carpeta uploads
    upload_folder = Path(app.config["UPLOAD_FOLDER"])
    upload_folder.mkdir(parents=True, exist_ok=True)

    # Importa aquí (después de crear app) para evitar import circular
    from .routes import web_bp, api_bp

    # Registra los blueprints
    app.register_blueprint(web_bp)
    app.register_blueprint(api_bp, url_prefix="/api")

    # app.config['DB_PATH'] = app.instance_path + '/embeddings.db'
    # app.embedding_db = EmbeddingDB(app.config['DB_PATH'])


    return app
