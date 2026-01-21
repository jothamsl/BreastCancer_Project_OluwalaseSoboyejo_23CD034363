"""
Breast Cancer Prediction System - Flask Web Application

Student Name: Oluwalase Soboyejo
Matric Number: 23CD034363

This application provides a web interface for predicting whether a breast tumor
is benign or malignant based on selected features.

Note: This system is strictly for educational purposes and must not be
presented as a medical diagnostic tool.
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'breast_cancer_model.pkl')

def load_model():
    """Load the trained model and its components."""
    try:
        components = joblib.load(MODEL_PATH)
        return components
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Load model at startup
model_components = load_model()

@app.route('/')
def home():
    """Render the home page with the prediction form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        # Check if model is loaded
        if model_components is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please ensure the model file exists.'
            })

        # Extract model components
        model = model_components['model']
        scaler = model_components['scaler']
        feature_names = model_components['feature_names']

        # Get input values from form
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form

        # Extract feature values
        radius_mean = float(data.get('radius_mean', 0))
        texture_mean = float(data.get('texture_mean', 0))
        perimeter_mean = float(data.get('perimeter_mean', 0))
        area_mean = float(data.get('area_mean', 0))
        concavity_mean = float(data.get('concavity_mean', 0))

        # Create feature array in the correct order
        features = np.array([[
            radius_mean,
            texture_mean,
            perimeter_mean,
            area_mean,
            concavity_mean
        ]])

        # Scale the features
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)[0]

        # Get prediction probabilities if available
        try:
            probabilities = model.predict_proba(features_scaled)[0]
            confidence = max(probabilities) * 100
        except:
            confidence = None

        # Determine result
        if prediction == 0:
            result = 'Malignant'
            result_class = 'malignant'
            description = 'The tumor is predicted to be MALIGNANT (cancerous).'
        else:
            result = 'Benign'
            result_class = 'benign'
            description = 'The tumor is predicted to be BENIGN (non-cancerous).'

        # Prepare response
        response = {
            'success': True,
            'prediction': result,
            'result_class': result_class,
            'description': description,
            'input_features': {
                'Radius Mean': radius_mean,
                'Texture Mean': texture_mean,
                'Perimeter Mean': perimeter_mean,
                'Area Mean': area_mean,
                'Concavity Mean': concavity_mean
            }
        }

        if confidence is not None:
            response['confidence'] = round(confidence, 2)

        # Check if it's an AJAX request
        if request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify(response)

        # For regular form submission, render template with results
        return render_template('index.html', result=response)

    except ValueError as e:
        error_response = {
            'success': False,
            'error': f'Invalid input value: {str(e)}. Please ensure all fields contain valid numbers.'
        }
        if request.is_json:
            return jsonify(error_response)
        return render_template('index.html', error=error_response['error'])

    except Exception as e:
        error_response = {
            'success': False,
            'error': f'An error occurred: {str(e)}'
        }
        if request.is_json:
            return jsonify(error_response)
        return render_template('index.html', error=error_response['error'])

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for prediction (JSON only)."""
    return predict()

@app.route('/health')
def health():
    """Health check endpoint."""
    model_status = 'loaded' if model_components is not None else 'not loaded'
    return jsonify({
        'status': 'healthy',
        'model_status': model_status
    })

@app.route('/about')
def about():
    """About page with information about the project."""
    return render_template('index.html', show_about=True)

# Error handlers
@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return render_template('index.html', error='Page not found.'), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    return render_template('index.html', error='Internal server error.'), 500

if __name__ == '__main__':
    # Check if model exists
    if model_components is None:
        print("=" * 60)
        print("WARNING: Model not found!")
        print("Please run model_building.ipynb first to create the model.")
        print("=" * 60)
    else:
        print("=" * 60)
        print("Breast Cancer Prediction System")
        print("Model loaded successfully!")
        print(f"Algorithm: K-Nearest Neighbors (K={model_components.get('best_k', 'N/A')})")
        print(f"Features: {model_components.get('feature_names', [])}")
        print("=" * 60)

    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
