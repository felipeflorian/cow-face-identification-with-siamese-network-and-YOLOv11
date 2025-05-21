from flask import Flask
from pathlib import Path

def create_app():
    app = Flask(__name__, instance_relative_config=False)
    app.config.from_object("config.Config")

    # Asegúrate de que exista la carpeta uploads
    upload_folder = Path(app.config["UPLOAD_FOLDER"])
    upload_folder.mkdir(parents=True, exist_ok=True)

    # Importa aquí (después de crear app) para evitar import circular
    from .routes import web_bp, api_bp

    # Registra los blueprints
    app.register_blueprint(web_bp)
    app.register_blueprint(api_bp, url_prefix="/api")

    return app
