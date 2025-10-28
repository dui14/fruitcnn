from flask import Flask
from app.config.config import Config
import app.views as bp

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    bp.init_app(app)
    return app
