from flask import Flask
from app.config.config import Config
import app.views as bp
import tensorflow as tf
import numpy as np
import os

UPLOAD_FOLDER = 'upload'



def create_app():
    app = Flask(__name__)

    os.makedirs(UPLOAD_FOLDER, exist_ok = True)
    app.config.from_object(Config)
    bp.init_app(app)
    return app
