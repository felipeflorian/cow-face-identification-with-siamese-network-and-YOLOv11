import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
from flask import (Blueprint, current_app, flash, jsonify, redirect,
                   render_template, request, url_for)
from PIL import Image

from .detector import DummyDetector, YoloDetector
from .recognizer import DummyRecognizer
from .utils.utils import allowed_file, draw_bboxes
from .pipeline import CowIDPipeline

web_bp = Blueprint("web", __name__)
api_bp = Blueprint("api", __name__)


# ——— Instancia global (lazy) del pipeline ———
def get_pipeline():
    # lo guardamos en current_app para no recrearlo en cada petición
    if not hasattr(current_app, "cow_pipeline"):
        current_app.cow_pipeline = CowIDPipeline(
            yolo_weights       = current_app.config["YOLO_WEIGHTS"],
            embedder_weights   = current_app.config["EMBEDDER_WEIGHTS"],
            gallery_pth        = current_app.config["GALLERY_PTH"],
            threshold          = current_app.config["EMBEDDING_THRESHOLD"],
            device             = current_app.config.get("DEVICE", None)
        )
    return current_app.cow_pipeline


#  Pagina web ---------------------------

@web_bp.route("/")
def index():
    return render_template("index.html")



@web_bp.route("/identify", methods=["POST"])
def identify_web():
    files = request.files.getlist("images")
    if not files:
        flash("Debes subir al menos una imagen", "danger")
        return redirect(url_for("web.index"))

    pipeline = get_pipeline()
    previews = []
    all_results = []

    for f in files:
        if not (f and allowed_file(f.filename)):
            continue
        img = Image.open(f.stream).convert("RGB")
        # procesar
        results = pipeline.identify(img)
        # guardar preview
        out_name = f"{uuid.uuid4().hex}.jpg"
        out_path = Path(current_app.config["UPLOAD_FOLDER"]) / out_name
        draw_bboxes(img.copy(), results, label_key="label").save(out_path)
        previews.append(url_for("static", filename=f"uploads/{out_name}"))
        all_results.append(results)

    if not previews:
        flash("No se procesó ninguna imagen válida", "warning")
        return redirect(url_for("web.index"))

    # empareja cada preview con sus resultados
    cards = [{"url":u, "results":r} for u, r in zip(previews, all_results)]

    return render_template("index.html",
        identified = True,
        cards      = cards
    )





# @web_bp.route("/identify", methods=["POST"])
# def identify_web():
#     embeddings_db = current_app.embedding_db.get_all_embeddings()

#     file = request.files.get("image")
#     if not (file and allowed_file(file.filename)):
#         flash("Imagen no válida", "danger")
#         return redirect(url_for("web.index"))

#     img = Image.open(file.stream).convert("RGB")
#     dets = detector.predict(img)
#     results = []
#     for d in dets:
#         #d["cow_id"] = cow_id
#         x, y, w, h = d["bbox"]
#         crop = img.crop((x, y, x+w, y+h))
#         emb = recognizer.encode(crop)

#         found_id, dist = None, float("inf")
#         for cow_id, ref in embeddings_db.items():
#             d0 = np.linalg.norm(emb - ref)
#             if d0 < dist:
#                 dist, found_id = d0, cow_id

#         if dist > current_app.config["EMBEDDING_THRESHOLD"]:
#             found_id = "Desconocida"

#         d["cow_id"] = found_id
#         results.append(d)

#     out_path = Path(current_app.config["UPLOAD_FOLDER"]) / f"{uuid.uuid4()}.jpg"
#     draw_bboxes(img, results).save(out_path)

#     return render_template("index.html",
#                            identified=True,
#                            image_url=url_for('static',
#                                              filename=f"uploads/{out_path.name}"),
#                            results=results)



@web_bp.route("/register", methods=["GET", "POST"])
def register():
    pipeline = get_pipeline()
    previews = []

    if request.method == "POST":
        cow_id = request.form.get("cow_id", "").strip()
        file   = request.files.get("image")  # ahora recibimos solo uno

        # validaciones
        if not cow_id or not (file and allowed_file(file.filename)):
            flash("Hace falta un ID o una imagen válida", "danger")
            return render_template("register.html", previews=previews)

        img = Image.open(file.stream).convert("RGB")

        # intentamos registrar en la galería
        ok = pipeline.register(img, cow_id)
        if not ok:
            flash("No se detectó ningún rostro de vaca", "warning")
            return render_template("register.html", previews=previews)

        # para previsualizar, dibujamos la caja detectada sobre la imagen
        dets = pipeline.identify(img)
        # suponemos que toma la primera detección
        bbox = dets[0]["bbox"]
        vis  = draw_bboxes(img.copy(), [{"bbox": bbox, "cow_id": cow_id}])
        filename = f"reg_{cow_id}_{uuid.uuid4().hex}.jpg"
        out_path = Path(current_app.config["UPLOAD_FOLDER"]) / filename
        vis.save(out_path)

        previews = [ url_for("static", filename=f"uploads/{filename}") ]
        flash(f"Vaca “{cow_id}” registrada correctamente", "success")

    return render_template("register.html", previews=previews)





