from flask import Flask
from app.config.config import Config

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # register blueprints of api routes
    from app.routes.ui.index_route import index_bp
    
    # register blueprints of ui routes
    app.register_blueprint(index_bp)

    return app
