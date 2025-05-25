# app/pipeline.py
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from .detector import YoloDetector
from .recognizer import CowRecognizer
import numpy as np
from .inference import predict_image  # tu función predict_image
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


class CowIDPipeline:
    def __init__(
        self,
        yolo_weights: str,
        embedder_weights: str,
        gallery_pth: str,
        threshold: float,
        device: str = None
    ):
        # 1) Device
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # 2) Detector
        self.detector = YoloDetector(weights=yolo_weights, conf=0.5, img_size=640)
        
        # 3) Embedder
        self.recognizer = CowRecognizer(weights=embedder_weights)
        self.recognizer.model.to(self.device)
        
        # 4) Galería pre-cargada
        gallery_embs, gallery_labels = torch.load(
            gallery_pth,
            map_location=self.device,
            weights_only=False
        )
        self.gallery_embs   = gallery_embs.to(self.device)
        self.gallery_labels = gallery_labels

        self.target_size = (100,150)
        
        self.gallery_pth = gallery_pth
        
        # 5) Transformación idéntica a la de eval en entrenamiento
        # self.transform = transforms.Compose([
        #     transforms.Resize(256),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        #     transforms.Normalize(
        #         mean=(0.485, 0.456, 0.406),
        #         std=(0.229, 0.224, 0.225)
        #     ),
        # ])

        self.transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
        
        # 6) Umbral open-set
        self.threshold = threshold

    def identify(self, img: Image.Image):
        """
        Dada una PIL.Image, devuelve lista de dicts:
        [{ 'bbox':[x,y,w,h],
        'label': cow_id,
        'distance': float,
        'det_confidence': float    # <-- añadimos esto
        }, …]
        """
        img.convert("RGB")
        dets = self.detector.predict(img)
        results = []
        for d in dets:
            x, y, w, h = d["bbox"]
            det_conf = d.get("confidence", None)          # confianza de YOLO
            face = img.crop((x, y, x+w, y+h))
            #face = face.resize(self.target_size)
            tensor = self.transform(face)
            label, dist = predict_image(
                tensor,
                self.recognizer.model,
                self.gallery_embs,
                self.gallery_labels,
                self.device,
                self.threshold
            )
            results.append({
                "bbox": d["bbox"],
                "label": label,
                "distance": dist,
                "det_confidence": det_conf,
            })
        return results
    

    
    def register(self, img: Image.Image, cow_id: str) -> bool:
            """
            Detecta la cara de la vaca en `img`, genera su embedding,
            lo añade a la galería y lo guarda en disco.
            Devuelve True si todo OK, False si no detectó nada.
            """
            dets = self.detector.predict(img)
            print (f"Encontradas {len(dets)} detecciones")

            if not dets:
                return False

            # 1) Crop y preprocesado idéntico al identify
            x, y, w, h = dets[0]["bbox"]
            face = img.crop((x, y, x + w, y + h))
            tensor = self.transform(face).unsqueeze(0).to(self.device)  # [1,C,H,W]

            # 2) Embedding
            with torch.no_grad():
                emb = F.normalize(self.recognizer.model(tensor), dim=1).cpu()  # [1, D]

            # 3) Append en memoria
            self.gallery_embs = torch.cat([self.gallery_embs, emb], dim=0)  # [G+1, D]
            self.gallery_labels = np.concatenate([self.gallery_labels, [cow_id]])  # [G+1]

            # 4) Persistir de vuelta
            torch.save((self.gallery_embs, self.gallery_labels), self.gallery_pth)

            return True

