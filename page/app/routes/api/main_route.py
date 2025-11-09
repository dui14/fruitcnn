from flask import Blueprint, jsonify

main_bp = Blueprint('main', __name__)   

@main_bp.route('/main', methods=['GET'])
def main():
    return jsonify(message="Welcome to the Main Route")