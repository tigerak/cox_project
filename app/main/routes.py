# Flask
from flask import request, jsonify
import requests
# modules
from config import *
from app.main import bp
from main import SmartAssistant

@bp.route('/api', methods=['POST'])
def api():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    # 데이터 처리
    assistant = SmartAssistant()
    response = assistant.process(data)

    return jsonify(response)