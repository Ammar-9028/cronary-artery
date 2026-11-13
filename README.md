# CardioScan AI - Coronary Artery Disease detection (Flask + MobileNetV2 + Grad-CAM)

## Local run (demo mode, no model needed)
    python3 -m venv .venv && . .venv/bin/activate
    pip install flask pillow numpy pytest
    pytest -q && python app.py        # http://localhost:5000

## Train a real model
Dataset folders: data/{train,val,test}/{negative,positive}/
    pip install -r requirements.txt -r requirements-train.txt
    python train.py --data data       # writes models/cad_model.keras, meta.json, metrics.json
Run `python app.py` again: DEMO banner disappears and Grad-CAM heatmaps appear.

## Docker / CI-CD
    docker build -t cad-detector . && docker run -p 5000:5000 -v $PWD/models:/app/models cad-detector
Jenkinsfile: checkout -> pytest -> docker build -> deploy (container `cad-app`, models mounted from /opt/cad/models).

Educational project, not a medical device.
