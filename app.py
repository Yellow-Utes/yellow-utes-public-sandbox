from flask import Flask, render_template, jsonify, redirect, request
import os
import logging

# Configure logging
from YellowUtesPredictor import YellowUtesPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Create Flask app
app = Flask(__name__)
#
# @app.before_request
# def enforce_https():
#     if request.headers.get('X-Forwarded-Proto', 'http') != 'https':
#         url = request.url.replace('http://', 'https://', 1)
#         return redirect(url, code=301)


@app.route('/')
def home():
    """Home page route"""
    return render_template('index.html')


@app.route('/health')
def health_check():
    """Health check endpoint for load balancer"""
    return jsonify({
        'status': 'healthy',
        'message': 'Application is running'
    })


@app.route('/api/inference')
def inference():

    predictor = YellowUtesPredictor(
        model_path="./models/current/timeseries_model.pth",
        config_path="./models/current/model_config.json",
        scaler_path="./models/current/model_scalers.pth"
    )
    results = predictor.predict_from_csv()
    return jsonify(results)


@app.route('/api/info')
def app_info():

    predictor = YellowUtesPredictor(
        model_path="./models/current/timeseries_model.pth",
        config_path="./models/current/model_config.json",
        scaler_path="./models/current/model_scalers.pth"
    )
    results = predictor.predict_from_csv()
    return jsonify(results)


@app.errorhandler(404)
def not_found(error):
    """404 error handler"""
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(error):
    """500 error handler"""
    logger.error(f"Internal server error: {error}")
    return render_template('500.html'), 500


# if __name__ == '__main__':
#     # This is used when running locally only
#     print("hellopw")
#     app.run(debug=True, host='0.0.0.0', port=5000)


# For Elastic Beanstalk, the application object must be called 'application'
application = app
