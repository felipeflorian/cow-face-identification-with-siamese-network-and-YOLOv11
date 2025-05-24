# # app/utils/db.py
# import sqlite3, pickle
# from typing import Dict
# import numpy as np

# # Helper para almacenar y recuperar los embeddings de vacas en una base de datos SQLite.
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
