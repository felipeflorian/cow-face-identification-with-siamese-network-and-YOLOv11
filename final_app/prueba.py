import app.routes
print("routes.py cargado desde:", app.routes.__file__)
print("Símbolos disponibles en app.routes:", [n for n in dir(app.routes) if not n.startswith("_")])

from app.routes import web_bp