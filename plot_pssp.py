import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

AA_TO_INT = {aa: i + 1 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
STRUCTURE_LABELS = ["H", "E", "C", "_"]
MAX_LEN = 128

model = tf.keras.models.load_model("models/PSSP.keras")


def encode(seq):
    indices = [AA_TO_INT.get(aa, 0) for aa in seq]
    indices = indices[:MAX_LEN]
    padded = np.zeros(MAX_LEN, dtype=int)
    padded[:len(indices)] = indices
    return padded


def probs_to_seq(probs):
    s = ""
    for p in probs:
        i = np.argmax(p)
        if i != 3:
            s += STRUCTURE_LABELS[i]
        else:
            break
    return s


EXAMPLE_SEQUENCES = [
    "TEFSEEQKRTLDLLFLFDRRMTEERRRWLSQRLGLNEEQIERWFRRKEQQI",
    "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR",
    "MKEEKRSSTGFLVKQRAFLKLYMITMTEQERLYGLKLLKVLQSEFKEIGFKPNHTEVYRSLHELLDDGILKQIKVKKEGAKLQEVVLYQFKDYEAAKLYKKQLKVELDRCKKLIEKALSDNF",
    "YCQKWMWTCDEERKCCEGLVCRLWCKRIINM",
    "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN",
    "PETYSNPETYSNPETYSNPETYSNPETYSNPETYSNPETYSNPETYSNPETYSNPETYSNPETYSNPETYSNPETYSNPETYSN",
]

for i, seq in enumerate(EXAMPLE_SEQUENCES):
    x = encode(seq)
    y_pred = model.predict(np.expand_dims(x, 0), verbose=0)[0]
    n = min(len(seq), MAX_LEN)
    y_pred = y_pred[:n]
    pred_str = probs_to_seq(y_pred).upper()
    conf = np.max(y_pred, axis=-1)

    print(f"\nSample {i+1}")
    print(f"Input : {seq[:n]}")
    print(f"Struct: {pred_str}")
    print(f"Conf  : {conf.mean():.1%} (mean)")

    fig, ax = plt.subplots(figsize=(max(10, n * 0.2), 2))
    ax.imshow(y_pred.T, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_yticks(range(len(STRUCTURE_LABELS)))
    ax.set_yticklabels(STRUCTURE_LABELS)
    ax.set_xlabel("Position")
    ax.set_title(f"Prédiction secondaire - Échantillon {i+1}")
    fig.tight_layout()
    fig.savefig(f"prediction_{i+1}.png", dpi=120, bbox_inches="tight")
    plt.show()
