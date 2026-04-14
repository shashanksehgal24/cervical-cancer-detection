from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

app = Flask(__name__)

# -----------------------------
# Load Model (MobileNetV3)
# -----------------------------
model = models.mobilenet_v3_small(pretrained=False)

# Modify final layer for 4 classes
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 4)

# Load trained weights (state_dict)
state_dict = torch.load("cervical_model.pth", map_location=torch.device("cpu"))
model.load_state_dict(state_dict)

model.eval()

# Class labels
classes = ["HSIL", "LSIL", "NILM", "SCC"]

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    try:
        image = Image.open(file).convert('RGB')
    except:
        return jsonify({"error": "Invalid image"}), 400

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    confidence, predicted = torch.max(probabilities, 0)

    confidence_value = float(confidence.item()) * 100
    predicted_class = classes[predicted.item()]

    # Confidence threshold logic
    if confidence_value < 90:
        predicted_class = "Uncertain"

    return jsonify({
        "class": predicted_class,
        "confidence": round(confidence_value, 2)
    })


# -----------------------------
# Run (for local testing only)
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)
