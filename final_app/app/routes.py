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
from .utils import allowed_file, draw_bboxes

web_bp = Blueprint("web", __name__)
api_bp = Blueprint("api", __name__)

detector = YoloDetector(conf=0.5, img_size=640)
recognizer = DummyRecognizer()
embeddings_db: Dict[str, np.ndarray] = {}   # Embeddings de vacas registradas dummy de prueba

#  Pagina web ---------------------------

@web_bp.route("/")
def index():
    return render_template("index.html")


@web_bp.route("/register", methods=["GET", "POST"])
def register():
    previews = []
    if request.method == "POST":
        files  = request.files.getlist("images")
        cow_id = request.form.get("cow_id", "").strip()

        if not (files and cow_id):
            flash("Falta imagen o ID", "danger")
            return render_template("register.html", previews=previews)

        added = 0
        for f in files:
            if not allowed_file(f.filename):
                continue

            img = Image.open(f.stream).convert("RGB")
            dets = detector.predict(img)
            if not dets:
                continue

            for d in dets:
                d["cow_id"] = cow_id

            # dibuja cajas en una copia
            vis = img.copy()
            vis = draw_bboxes(vis, dets)

            # guarda la imagen con bounding boxes
            filename = f"{cow_id}_{added}_{uuid.uuid4().hex}.jpg"
            out_path = Path(current_app.config["UPLOAD_FOLDER"]) / filename
            vis.save(out_path)

            # guarda embedding
            x, y, w, h = dets[0]["bbox"]
            emb = recognizer.encode(img.crop((x, y, x+w, y+h)))
            embeddings_db[f"{cow_id}_{added}"] = emb

            # añade la url para previsualizar
            previews.append(url_for('static', filename=f"uploads/{filename}"))
            added += 1

        if added == 0:
            flash("No se pudo procesar ninguna imagen", "warning")
        else:
            flash(f"Registradas {added} imagen(es) para la vaca con id {cow_id}", "success")

    return render_template("register.html", previews=previews)




@web_bp.route("/identify", methods=["POST"])
def identify_web():
    file = request.files.get("image")
    if not (file and allowed_file(file.filename)):
        flash("Imagen no válida", "danger")
        return redirect(url_for("web.index"))

    img = Image.open(file.stream).convert("RGB")
    dets = detector.predict(img)
    results = []
    for d in dets:
        #d["cow_id"] = cow_id
        x, y, w, h = d["bbox"]
        crop = img.crop((x, y, x+w, y+h))
        emb = recognizer.encode(crop)

        found_id, dist = None, float("inf")
        for cow_id, ref in embeddings_db.items():
            d0 = np.linalg.norm(emb - ref)
            if d0 < dist:
                dist, found_id = d0, cow_id

        if dist > current_app.config["EMBEDDING_THRESHOLD"]:
            found_id = "Desconocida"

        d["cow_id"] = found_id
        results.append(d)

    out_path = Path(current_app.config["UPLOAD_FOLDER"]) / f"{uuid.uuid4()}.jpg"
    draw_bboxes(img, results).save(out_path)

    return render_template("index.html",
                           identified=True,
                           image_url=url_for('static',
                                             filename=f"uploads/{out_path.name}"),
                           results=results)




# Rutas API -------------------------------- 

@api_bp.route("/register/<cow_id>", methods=["POST"])
def register_api(cow_id):
    file = request.files.get("image")
    if not (file and allowed_file(file.filename)):
        return {"error": "image required"}, 400
    img = Image.open(file.stream).convert("RGB")
    dets = detector.predict(img)
    if not dets:
        return {"error": "no cow head detected"}, 422
    x, y, w, h = dets[0]["bbox"]
    emb = recognizer.encode(img.crop((x, y, x+w, y+h)))
    embeddings_db[cow_id] = emb

    print(embeddings_db.get(cow_id))
    return {"cow_id": cow_id, "embedding_size": len(emb)}



@api_bp.route("/identify", methods=["POST"])
def identify_api():
    file = request.files.get("image")
    if not (file and allowed_file(file.filename)):
        return {"error": "image required"}, 400
¿


