from flask import Flask, request, jsonify, render_template
import torch
from torchvision import transforms
from PIL import Image

app = Flask(__name__)

# Load model once (important for performance)
model = torch.load("cervical_model.pth", map_location=torch.device('cpu'))
model.eval()

classes = ["HSIL", "LSIL", "NILM", "SCC"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    image = Image.open(file).convert('RGB')

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    confidence, predicted = torch.max(probabilities, 0)

    result = {
        "class": classes[predicted.item()],
        "confidence": float(confidence.item()) * 100
    }

    return jsonify(result)
