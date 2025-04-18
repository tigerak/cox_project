from flask import Flask
from flask_cors import CORS
# modules
import config as config


def create_app():
    app = Flask(__name__)
    app.config.from_object(config)
    
    # CORS 설정 추가
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    from app.main import bp
    app.register_blueprint(bp)
    
    return app

app = create_app()
    
if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)