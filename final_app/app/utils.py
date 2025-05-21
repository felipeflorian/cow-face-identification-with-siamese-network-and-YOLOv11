from PIL import Image, ImageDraw, ImageFont
import importlib.resources as pkg_resources

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

def allowed_file(fname: str) -> bool:
    return "." in fname and fname.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def draw_bboxes(img: Image.Image, detections):
    draw = ImageDraw.Draw(img)
    # Usa fuente por defecto de PIL
    for d in detections:
        x, y, w, h = d["bbox"]
        draw.rectangle([(x, y), (x+w, y+h)], outline="red", width=3)
        draw.text((x, y-10), d["cow_id"], fill="red")
    return img
