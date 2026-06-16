import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from io import BytesIO
import base64

from config import MODEL_PATH, MAX_LEN, AA_TO_INT, STRUCTURE_LABELS

_model = None

def get_model():
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model

def encode_sequence(seq):
    indices = [AA_TO_INT.get(aa, 0) for aa in seq]
    indices = indices[:MAX_LEN]
    padded = np.zeros(MAX_LEN, dtype=int)
    padded[:len(indices)] = indices
    return padded

def decode_prediction(oh_seq):
    s = ""
    for o in oh_seq:
        i = np.argmax(o)
        if i < len(STRUCTURE_LABELS) - 1:
            s += STRUCTURE_LABELS[i]
        else:
            break
    return s

def predict(seq):
    model = get_model()
    x = encode_sequence(seq)
    x_batch = np.expand_dims(x, 0)
    y_pred = model.predict(x_batch, verbose=0)[0]
    n = min(len(seq), MAX_LEN)
    probs = y_pred[:n]
    pred_seq = decode_prediction(probs)
    return pred_seq, probs

def plot_prediction(seq, probs):
    n = len(seq)
    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor("#13151F")
    ax.set_facecolor("#13151F")
    ax.tick_params(colors="#8B8FA8")
    for spine in ax.spines.values():
        spine.set_color((1, 1, 1, 0.07))

    probs_t = probs.T[:3]
    extent = [-0.5, n - 0.5, -0.5, 2.5]
    im = ax.imshow(probs_t, cmap="Reds", aspect="auto", extent=extent, vmin=0, vmax=1, origin="lower")
    ax.set_yticks(range(3))
    visible_labels = STRUCTURE_LABELS[:3]
    ax.set_yticklabels(visible_labels, fontfamily="monospace", fontsize=10, color="#F0EEF8")
    ax.set_xlabel("Position", color="#8B8FA8", fontsize=9)
    step = max(1, n // 20)
    ax.set_xticks(range(0, n, step))
    ax.tick_params(axis="x", colors="#8B8FA8", labelsize=8)

    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor="#13151F", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode()
