#!/usr/bin/env python3
"""
Web application for dental diagnosis using LoRA fine-tuned model
"""

from flask import Flask, render_template, request, jsonify
import os
import base64
from pathlib import Path
from werkzeug.utils import secure_filename
import tempfile
import shutil

# Import our model
from core.llava_lora_model import LLaVALoRAModel
from core.model_configs import get_model_config

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_EXTENSIONS'] = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}

# Initialize model globally (loaded once)
model = None

def load_model():
    """Load the fine-tuned model"""
    global model
    if model is None:
        print("Loading fine-tuned model...")
        config = get_model_config("tiny-llava")
        model = LLaVALoRAModel(
            model_name=config["model_name"],
            lora_config=config.get("lora_config"),
            use_quantization=config.get("use_quantization", False),
            quantization_bits=config.get("quantization_bits", 4)
        )

        # Load fine-tuned LoRA weights if available
        checkpoint_path = Path("checkpoints/lora_adapters/final")
        if checkpoint_path.exists():
            model.load_lora_adapter(str(checkpoint_path))
            print("✓ Loaded fine-tuned LoRA weights")
        else:
            print("⚠ Using base model (no fine-tuned weights found)")

    return model

@app.route('/')
def index():
    """Home page with upload form"""
    return render_template('index.html')

@app.route('/diagnose', methods=['POST'])
def diagnose():
    """Handle image upload and return diagnosis"""
    try:
        # Check if file was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Validate file extension
        filename = secure_filename(file.filename)
        file_ext = Path(filename).suffix.lower()

        if file_ext not in app.config['UPLOAD_EXTENSIONS']:
            return jsonify({'error': f'Invalid file type. Allowed: {", ".join(app.config["UPLOAD_EXTENSIONS"])}'}), 400

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            file.save(tmp_file.name)
            temp_path = tmp_file.name

        try:
            # Load model if needed
            model = load_model()

            # Generate diagnosis
            print(f"Generating diagnosis for: {filename}")
            diagnosis = model.generate_diagnosis(temp_path)

            # Encode image to base64 for display
            with open(temp_path, 'rb') as img_file:
                image_data = base64.b64encode(img_file.read()).decode('utf-8')

            # Clean up temp file
            os.unlink(temp_path)

            # Parse diagnosis into structured format
            lines = diagnosis.strip().split('\n')
            structured_diagnosis = {
                'full_text': diagnosis,
                'summary': lines[0] if lines else "Analysis complete",
                'details': []
            }

            # Extract key points from diagnosis
            for line in lines:
                line = line.strip()
                if line and any(keyword in line.lower() for keyword in ['diagnosis', 'severity', 'condition', 'treatment', 'urgency', 'recommendation']):
                    structured_diagnosis['details'].append(line)

            return jsonify({
                'success': True,
                'diagnosis': structured_diagnosis,
                'image': f'data:image/{file_ext[1:]};base64,{image_data}',
                'filename': filename
            })

        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise e

    except Exception as e:
        print(f"Error during diagnosis: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    # Pre-load model on startup
    print("Starting Dental Diagnosis Web App...")
    load_model()

    # Run Flask app
    app.run(host='0.0.0.0', port=5001, debug=False)