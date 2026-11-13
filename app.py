import io, os
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError
from cad_model import Predictor

ALLOWED = {"png", "jpg", "jpeg", "bmp", "webp"}


def create_app(model_path=None):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB
    predictor = Predictor(model_path or os.getenv("MODEL_PATH", "models/cad_model.keras"))

    @app.get("/")
    def index():
        return render_template("index.html", demo=predictor.demo)

    @app.get("/health")
    def health():
        return jsonify(status="ok", mode="demo" if predictor.demo else "model")

    @app.post("/predict")
    def predict():
        f = request.files.get("image")
        if f is None or not f.filename:
            return jsonify(error="Image select karo (field name: image)."), 400
        if f.filename.rsplit(".", 1)[-1].lower() not in ALLOWED:
            return jsonify(error="Sirf PNG, JPG, BMP ya WEBP image allowed hai."), 400
        raw = f.read()
        try:
            img = Image.open(io.BytesIO(raw))
            img.load()
        except (UnidentifiedImageError, OSError):
            return jsonify(error="Yeh valid image file nahi hai."), 400
        return jsonify(predictor.predict(img, raw))

    @app.errorhandler(413)
    def too_big(_):
        return jsonify(error="File 8 MB se badi hai."), 413

    @app.after_request
    def headers(r):
        r.headers["X-Content-Type-Options"] = "nosniff"
        r.headers["X-Frame-Options"] = "DENY"
        r.headers["Cache-Control"] = "no-store" if request.path == "/predict" else "public, max-age=300"
        return r

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
