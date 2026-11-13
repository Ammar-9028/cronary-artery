"""Train the CAD classifier (MobileNetV2 transfer learning).

Dataset layout (folder names matter: negative=0, positive=1):
  data/train/{negative,positive}/*.png|jpg
  data/val/{negative,positive}/
  data/test/{negative,positive}/
Run:  python train.py --data data --epochs 8 --ft-epochs 10
Outputs: models/cad_model.keras, models/meta.json, models/metrics.json
"""
import argparse, json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

IMG, CLASSES = 224, ["negative", "positive"]


def load(path, bs, shuffle):
    ds = tf.keras.utils.image_dataset_from_directory(
        path, label_mode="binary", class_names=CLASSES, image_size=(IMG, IMG),
        batch_size=bs, shuffle=shuffle, seed=42)
    return ds


def build():
    base = tf.keras.applications.MobileNetV2(input_shape=(IMG, IMG, 3), include_top=False, weights="imagenet")
    base.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(base.input, out), base   # flat graph -> layer 'out_relu' usable for Grad-CAM


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--data", default="data"); a.add_argument("--bs", type=int, default=32)
    a.add_argument("--epochs", type=int, default=8); a.add_argument("--ft-epochs", type=int, default=10)
    a = a.parse_args()

    aug = tf.keras.Sequential([layers.RandomFlip("horizontal"), layers.RandomRotation(0.05),
                               layers.RandomZoom(0.1), layers.RandomContrast(0.15)])
    norm = lambda x, y: (x / 127.5 - 1.0, y)
    train_raw, val_raw, test_raw = (load(f"{a.data}/{s}", a.bs, s == "train") for s in ("train", "val", "test"))
    ytr = np.concatenate([y.numpy() for _, y in train_raw]).ravel()
    n_neg, n_pos = int((ytr == 0).sum()), int((ytr == 1).sum())
    cw = {0: len(ytr) / (2 * n_neg), 1: len(ytr) / (2 * n_pos)}
    print("class counts", n_neg, n_pos, "weights", cw)
    AT = tf.data.AUTOTUNE
    train = train_raw.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AT).map(norm).prefetch(AT)
    val, test = val_raw.map(norm).prefetch(AT), test_raw.map(norm).prefetch(AT)

    model, base = build()
    cb = [tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=4, restore_best_weights=True),
          tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)]
    metrics = ["accuracy", tf.keras.metrics.AUC(name="auc")]
    model.compile(tf.keras.optimizers.Adam(1e-3), "binary_crossentropy", metrics=metrics)
    model.fit(train, validation_data=val, epochs=a.epochs, class_weight=cw, callbacks=cb)

    base.trainable = True
    for l in base.layers[:-30]:
        l.trainable = False
    model.compile(tf.keras.optimizers.Adam(1e-5), "binary_crossentropy", metrics=metrics)
    model.fit(train, validation_data=val, epochs=a.ft_epochs, class_weight=cw, callbacks=cb)

    # threshold from validation set (Youden's J), then final report on the untouched test set
    yv = np.concatenate([y.numpy() for _, y in val]).ravel(); pv = model.predict(val).ravel()
    fpr, tpr, thr = roc_curve(yv, pv); thr_best = float(thr[np.argmax(tpr - fpr)])
    yt = np.concatenate([y.numpy() for _, y in test]).ravel(); pt = model.predict(test).ravel()
    pred = (pt >= thr_best).astype(int)
    res = {"threshold": thr_best, "test_auc": float(roc_auc_score(yt, pt)),
           "confusion_matrix": confusion_matrix(yt, pred).tolist(),
           "report": classification_report(yt, pred, target_names=CLASSES, output_dict=True)}
    model.save("models/cad_model.keras")
    json.dump({"threshold": thr_best}, open("models/meta.json", "w"))
    json.dump(res, open("models/metrics.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
