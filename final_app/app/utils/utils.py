from PIL import Image, ImageDraw, ImageFont
# import importlib.resources as pkg_resources
# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader
# import os
# import pandas as pd




ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

def allowed_file(fname: str) -> bool:
    return "." in fname and fname.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def draw_bboxes(
    img: Image.Image,
    detections: list[dict],
    label_key: str = "cow_id"
) -> Image.Image:
    """
    Dibuja rectángulos y etiquetas sobre `img`.
    - detections: lista de dicts con al menos "bbox" y la clave de etiqueta.
    - label_key: nombre de la clave en cada dict que contiene la etiqueta a mostrar.
    """
    draw = ImageDraw.Draw(img)
    for d in detections:
        x, y, w, h = d["bbox"]
        # rectángulo
        draw.rectangle([(x, y), (x+w, y+h)], outline="red", width=3)
        # etiqueta (usa label_key, cae en vacío si no existe)
        label = d.get(label_key, "")
        draw.text((x, y-10), str(label), fill="red")
    return img


# class CowSingle(Dataset):
#     """Devuelve imagen, label  sin trípletas para  test."""
#     def __init__(self, df, transform=None):
#         self.imgs = df['filepath'].values
#         self.labs = df['label'].values
#         self.tf   = transform

#     def __len__(self): return len(self.imgs)

#     def __getitem__(self, i):
#         img = Image.open(self.imgs[i])
#         if self.tf: img = self.tf(img)
#         return img, self.labs[i]
    

# def get_gallery_dataset(gallery_df,  eval_transform):
#     gallery_dataset = CowSingle(df= gallery_df, transform=eval_transform) 

#     return gallery_dataset

# def get_face_dataloaders(gallery_dataset, batch_size=32, num_workers=4):
#     gallery_dataloader = DataLoader(gallery_dataset, batch_size=batch_size, shuffle=False)

#     return gallery_dataloader

# ## Pasar las imamgenes
# def get_df(path, minimo_numero_imagenes):
#   filepaths = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path)
#               for f in filenames if os.path.splitext(f)[1].lower() in ['.png','.jpg','.jpeg']]
#   labels = [os.path.basename(os.path.dirname(path)) for path in filepaths]
#   df = pd.DataFrame({'filepath': filepaths, 'label': labels})
#   # se filtran las clases con menos de 11 imagenes par ano tener problemas con la creacion de tripletas
#   df = df.groupby("label").filter(lambda grp: len(grp) > minimo_numero_imagenes)
#   # Se crea una galeria para simular un sistema en produccion 
#   gallery_df = (df.groupby('label', group_keys=False).apply(lambda x: x.sample(1, random_state=42)).reset_index(drop=True))
#   return df, gallery_df


# # app/utils/db.py
# import sqlite3, pickle

# #Importar anotaciones
# from typing import Dict
# import numpy as np

# class EmbeddingDB:
#     def __init__(self, db_path: str):
#         self.conn = sqlite3.connect(db_path, check_same_thread=False)
#         self.conn.execute("""
#           CREATE TABLE IF NOT EXISTS embeddings (
#             cow_id   TEXT PRIMARY KEY,
#             vector   BLOB
#           )
#         """)
#         self.conn.commit()

#     def add(self, cow_id: str, emb: np.ndarray):
#         blob = pickle.dumps(emb)
#         self.conn.execute(
#             "INSERT OR REPLACE INTO embeddings (cow_id, vector) VALUES (?, ?)",
#             (cow_id, blob)
#         )
#         self.conn.commit()

#     def all(self) -> Dict[str, np.ndarray]:
#         cur = self.conn.execute("SELECT cow_id, vector FROM embeddings")
#         return {
#           cow_id: pickle.loads(blob)
#           for cow_id, blob in cur.fetchall()
#         }

#     def close(self):
#         self.conn.close()

