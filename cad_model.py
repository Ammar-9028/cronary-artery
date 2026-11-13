"""Inference engine: loads the trained Keras model (if present) and produces
prediction + Grad-CAM heatmap. TensorFlow is imported lazily so the web app and
unit tests also run without it (DEMO mode)."""
import base64, hashlib, io, json, os, time
import numpy as np
from PIL import Image

IMG = 224


def _b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _colorize(cam: np.ndarray) -> np.ndarray:
    """Simple blue->cyan->yellow->red colormap, cam in [0,1]."""
    r = np.clip(1.5 - np.abs(4 * cam - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * cam - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * cam - 1), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype("uint8")


class Predictor:
    def __init__(self, model_path="models/cad_model.keras", meta_path="models/meta.json"):
        self.model = None
        self.threshold = 0.5
        self._gradcam_model = None
        if os.path.exists(meta_path):
            self.threshold = float(json.load(open(meta_path)).get("threshold", 0.5))
        if os.path.exists(model_path):
            import tensorflow as tf
            self.tf = tf
            self.model = tf.keras.models.load_model(model_path)
            try:
                self._gradcam_model = tf.keras.Model(
                    self.model.inputs,
                    [self.model.get_layer("out_relu").output, self.model.output])
            except Exception:
                self._gradcam_model = None

    @property
    def demo(self) -> bool:
        return self.model is None

    def _gradcam(self, x):
        if self._gradcam_model is None:
            return None
        tf = self.tf
        with tf.GradientTape() as tape:
            conv, pred = self._gradcam_model(x)
            score = pred[:, 0]
        grads = tape.gradient(score, conv)
        weights = tf.reduce_mean(grads, axis=(1, 2))
        cam = tf.reduce_sum(conv * weights[:, None, None, :], axis=-1)[0]
        cam = tf.nn.relu(cam).numpy()
        return cam / (cam.max() + 1e-8)

    def predict(self, img: Image.Image, raw: bytes) -> dict:
        t0 = time.time()
        img = img.convert("RGB").resize((IMG, IMG))
        arr = np.asarray(img, dtype="float32")
        heat = None
        if self.demo:
            # NOT a real prediction: deterministic pseudo-score so the UI can be demoed.
            prob = int(hashlib.sha256(raw).hexdigest()[:8], 16) / 0xFFFFFFFF
        else:
            x = arr[None] / 127.5 - 1.0
            prob = float(self.model.predict(x, verbose=0)[0][0])
            cam = self._gradcam(x)
            if cam is not None:
                cam_img = Image.fromarray((cam * 255).astype("uint8")).resize((IMG, IMG), Image.BICUBIC)
                cam_arr = np.asarray(cam_img, dtype="float32") / 255.0
                over = 0.55 * arr + 0.45 * _colorize(cam_arr)
                heat = _b64(Image.fromarray(over.astype("uint8")))
        positive = prob >= self.threshold
        return {
            "label": "Positive" if positive else "Negative",
            "positive": bool(positive),
            "probability": round(prob, 4),
            "confidence": round(prob if positive else 1 - prob, 4),
            "threshold": self.threshold,
            "demo": self.demo,
            "original": _b64(img),
            "heatmap": heat,
            "ms": int((time.time() - t0) * 1000),
        }
