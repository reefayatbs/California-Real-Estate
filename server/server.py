from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_swagger_ui import get_swaggerui_blueprint
from marshmallow import Schema, fields, ValidationError
import logging.config
import os
import util
from config import LOGGING_CONFIG, CORS_ORIGINS, RATE_LIMIT

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# Initialize artifacts
try:
    util.load_artifacts()
    logger.info("Artifacts loaded successfully")
except Exception as e:
    logger.error(f"Failed to load artifacts: {str(e)}")
    raise

app = Flask(__name__)

# Configure CORS
CORS(app, resources={
    r"/api/*": {
        "origins": CORS_ORIGINS,
        "methods": ["GET", "POST", "OPTIONS"]
    }
})

# Configure rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=[RATE_LIMIT]
)

# Input validation schemas
class PredictionRequestSchema(Schema):
    house_size = fields.Float(required=True, validate=lambda x: x > 0)
    city = fields.String(required=True)
    bed = fields.Integer(required=True, validate=lambda x: x >= 0)
    bath = fields.Integer(required=True, validate=lambda x: x >= 0)

prediction_schema = PredictionRequestSchema()

@app.route("/")
def home():
    return jsonify({
        "status": "success",
        "message": "California Real Estate Price Predictor API",
        "endpoints": {
            "cities": "/api/v1/cities",
            "predict": "/api/v1/predict"
        }
    })

@app.route("/api/v1/cities", methods=['GET'])
@limiter.limit(RATE_LIMIT)
def get_city_names():
    try:
        cities = util.get_city_names()
        logger.info(f"Retrieved {len(cities)} cities")
        return jsonify({
            'status': 'success',
            'cities': cities
        })
    except Exception as e:
        logger.error(f"Error retrieving cities: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Internal server error'
        }), 500

@app.route("/api/v1/predict", methods=['POST'])
@limiter.limit(RATE_LIMIT)
def predict_home_price():
    try:
        if not request.is_json:
            return jsonify({
                'status': 'error',
                'message': 'Content-Type must be application/json'
            }), 400

        # Validate input
        try:
            data = prediction_schema.load(request.get_json())
        except ValidationError as err:
            return jsonify({
                'status': 'error',
                'message': 'Validation error',
                'errors': err.messages
            }), 400

        # Log prediction request
        logger.info(f"Prediction request received for {data['city']}")

        # Get prediction
        estimated_price = util.get_estimated_price(
            data['city'],
            data['house_size'],
            data['bed'],
            data['bath']
        )

        logger.info(f"Prediction successful: ${estimated_price:,.2f}")
        
        return jsonify({
            'status': 'success',
            'estimated_price': estimated_price
        })

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Internal server error'
        }), 500

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        'status': 'error',
        'message': 'Rate limit exceeded'
    }), 429

if __name__ == "__main__":
    logger.info("Starting server")
    app.run(host='0.0.0.0', port=5001, debug=True)
