# topic_modeling_pipeline.py

#
# ─────────────────────────────────────────────────────────────────────────────
# Notation mapping symbols:
#   V           := vocabulary
#   E ∈ R^{|V|×d}  := embedding matrix (word → d-dim vector)
#   U           := sequence of utterances in a dialogue (length |U|)
#   T           := max tokens per utterance (sequence length at word-level)
#   H_t         := word-level BiLSTM hidden states for utterance tokens
#   s(u)        := utterance embedding (self-attentive pooling over H_t)
#   Θ(U)        := contextual state over the utterance sequence U
#                  (BiLSTM over [s(u1),…,s(u|U|)], shape |U|×2H_c)
#   Φ(U)        := per-utterance topic posteriors (TimeDistributed softmax),
#                  Φ(u_t) ∈ Δ^{|L|-1} over label/topic set L
#   L           := set of topics/labels; |L| topics
#   𝔅 (BoT)     := Bag-of-Topics prior (topic → keyword list)
#   K           := keywords; K_t are keywords of topic t ∈ L
#
# Evaluation:
#   MAE on segment counts, WindowDiff (Pevzner & Hearst, 2002), macro P/R/F1
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
import math
import argparse
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# Data I/O
# ─────────────────────────────────────────────────────────────────────────────
def read_dialogue_csv(path: str) -> pd.DataFrame:
    """
    Required columns: dialogue_id, turn_id, utterance
    Optional columns: gold_topic, gold_da
    """
    df = pd.read_csv(path)
    req = {"dialogue_id","turn_id","utterance"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    return df.sort_values(["dialogue_id","turn_id"]).reset_index(drop=True)

def read_bot_json(path: str) -> Dict[str, List[str]]:
    """𝔅 (BoT) : topic → keywords (lowercased, deduped)."""
    with open(path, "r", encoding="utf-8") as f:
        bot = json.load(f)
    return {t: sorted(set([w.lower().strip() for w in kws])) for t, kws in bot.items()}

def split_by_dialogue(df: pd.DataFrame, tr=0.8, dv=0.1, te=0.1):
    dids = df["dialogue_id"].unique().tolist()
    random.shuffle(dids)
    n = len(dids); n_tr = int(n*tr); n_dv = int(n*dv)
    tr_ids = set(dids[:n_tr])
    dv_ids = set(dids[n_tr:n_tr+n_dv])
    te_ids = set(dids[n_tr+n_dv:])
    return (df[df.dialogue_id.isin(tr_ids)].copy(),
            df[df.dialogue_id.isin(dv_ids)].copy(),
            df[df.dialogue_id.isin(te_ids)].copy())

# ─────────────────────────────────────────────────────────────────────────────
# Vocabulary & Embeddings (V, E)
# ─────────────────────────────────────────────────────────────────────────────
def build_vocab(texts: List[str], max_words=50000, lower=True) -> Tokenizer:
    """Tokenizer models V; OOV token <unk>."""
    Vocab = Tokenizer(num_words=max_words, lower=lower, oov_token="<unk>")
    Vocab.fit_on_texts(texts)
    return Vocab

def texts_to_UxT(Vocab: Tokenizer, texts: List[str], T_max=40) -> np.ndarray:
    """Convert utterances to padded token ids: shape (|U|, T)."""
    seqs = Vocab.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=T_max, padding="post", truncating="post")

def load_embeddings(path: str, word_index: Dict[str,int], d=300, max_words=50000) -> np.ndarray:
    """
    E ∈ R^{|V|×d}; if file is missing, random init (N(0, 0.02)).
    """
    V_size = min(max_words, len(word_index)+1)
    E = np.random.normal(scale=0.02, size=(V_size, d)).astype(np.float32)
    if path and os.path.exists(path):
        print(f"Loading embeddings from {path} ...")
        store = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split()
                if len(parts) != d+1:   # skip malformed lines
                    continue
                w = parts[0]; vec = np.asarray(parts[1:], dtype=np.float32)
                store[w] = vec
        hit = 0
        for w, i in word_index.items():
            if i < V_size and w in store:
                E[i] = store[w]; hit += 1
        print(f"Initialised {hit}/{V_size} tokens from pre-trained vectors.")
    else:
        print("No embedding file provided; using random initialisation for E.")
    return E

# ─────────────────────────────────────────────────────────────────────────────
# Noisy Labelling using 𝔅 (BoT)
# ─────────────────────────────────────────────────────────────────────────────
def keyword_vote(utt: str, BoT: Dict[str, List[str]]) -> Tuple[str, int]:
    """Count keyword hits per topic; return (argmax_topic, hits)."""
    toks = utt.lower().split()
    counts = {t: sum(tok in set(kws) for tok in toks) for t, kws in BoT.items()}
    t_hat = max(counts.items(), key=lambda x: x[1]) if counts else (None, 0)
    return t_hat[0], t_hat[1]

def utter_avg_vec(utt: str, Vocab: Tokenizer, E: np.ndarray, T_max_idx: int) -> np.ndarray:
    vecs = []
    for w in utt.lower().split():
        idx = Vocab.word_index.get(w, 0)
        if idx < E.shape[0] and idx <= T_max_idx:
            vecs.append(E[idx])
    if vecs:
        return np.mean(np.stack(vecs, axis=0), axis=0)
    return np.zeros((E.shape[1],), dtype=np.float32)

def build_topic_centroids(BoT: Dict[str, List[str]], word_index: Dict[str,int], E: np.ndarray) -> Dict[str, np.ndarray]:
    C = {}
    for t, kws in BoT.items():
        vecs = []
        for w in kws:
            idx = word_index.get(w, 0)
            if idx < E.shape[0]:
                vecs.append(E[idx])
        if vecs:
            C[t] = np.mean(np.stack(vecs, axis=0), axis=0)
    return C

def cos_sim(a: np.ndarray, b: np.ndarray, eps=1e-8) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < eps or nb < eps: return 0.0
    return float(np.dot(a,b) / (na*nb))

def noisy_label(df: pd.DataFrame,
                BoT: Dict[str, List[str]],
                Vocab: Tokenizer,
                E: np.ndarray) -> pd.DataFrame:
    """
    Step 1 (Keyword): assign argmax by keyword hits.
    Step 2 (1-NN): if no hits, assign nearest topic centroid in embedding space.
    """
    df = df.copy()
    centroids = build_topic_centroids(BoT, Vocab.word_index, E)
    T_max_idx = E.shape[0] - 1
    topics = sorted(BoT.keys())
    labels = []
    for utt in df["utterance"].astype(str):
        t_star, hits = keyword_vote(utt, BoT)
        if t_star and hits > 0:
            labels.append(t_star); continue
        u_vec = utter_avg_vec(utt, Vocab, E, T_max_idx)
        if not centroids:
            labels.append(random.choice(topics))
        else:
            sims = {t: cos_sim(u_vec, c) for t, c in centroids.items()}
            labels.append(max(sims.items(), key=lambda x: x[1])[0])
    df["noisy_topic"] = labels
    return df

# ─────────────────────────────────────────────────────────────────────────────
# Model: s(u) via word-level BiLSTM + self-attn; Θ(U) via context BiLSTM; Φ(U) via softmax
# ─────────────────────────────────────────────────────────────────────────────
class SelfAttention(layers.Layer):
    """Additive self-attention over time: returns s(u) and attention weights."""
    def __init__(self, a_units=128, **kwargs):
        super().__init__(**kwargs)
        self.W = layers.Dense(a_units, activation="tanh")
        self.v = layers.Dense(1)

    def call(self, H_t, mask=None, training=None):
        # H_t: (B, T, d_h)
        score = self.v(self.W(H_t))  # (B, T, 1)
        if mask is not None:
            mask = tf.cast(mask, tf.float32)       # (B, T)
            mask = tf.expand_dims(mask, axis=-1)   # (B, T, 1)
            score = score + (1.0 - mask) * (-1e9)  # large negative where masked
        A = tf.nn.softmax(score, axis=1)           # (B, T, 1)
        s_u = tf.reduce_sum(A * H_t, axis=1)       # s(u) ∈ R^{d_h}
        return s_u, tf.squeeze(A, axis=-1)

def build_segmenter(V, d, E, T_max, L, H_ctx=256) -> models.Model:
    """
    Word encoder (BiLSTM+attn) → s(u); Context BiLSTM over U → Θ(U); Dense→ Φ(U).
    """
    # Input: batch of dialogues, each is a sequence of utterances U, each utterance has T tokens
    U_tokens = layers.Input(shape=(None, T_max), dtype="int32", name="U_tokens")  # (B, |U|, T)

    # Word-level encoder shared for all utterances: H_t → s(u)
    token_ids = layers.Input(shape=(T_max,), dtype="int32")
    E_layer = layers.Embedding(V, d, weights=[E], trainable=True, mask_zero=True)(token_ids)
    H_t = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(E_layer)  # (B, T, 256)
    s_u, _ = SelfAttention(128)(H_t)                                             # (B, 256)
    utter_encoder = models.Model(token_ids, s_u, name="utter_encoder")

    # Time-distribute encoder over utterance axis to get [s(u1),…,s(u|U|)]
    S_U = layers.TimeDistributed(utter_encoder, name="encode_U")(U_tokens)        # (B, |U|, 256)

    # Θ(U): contextual BiLSTM over utterance sequence
    Theta_U = layers.Bidirectional(layers.LSTM(H_ctx, return_sequences=True), name="Theta")(S_U)  # (B, |U|, 2H_ctx)

    # Φ(U): per-utterance topic posterior (softmax over |L|)
    Phi_U = layers.TimeDistributed(layers.Dense(L, activation="softmax"), name="Phi")(Theta_U)  # (B, |U|, |L|)

    model = models.Model(U_tokens, Phi_U)
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

# ─────────────────────────────────────────────────────────────────────────────
# Packing dialogues: U×T tensors and per-utterance labels in L
# ─────────────────────────────────────────────────────────────────────────────
def pack_dialogues(df: pd.DataFrame,
                   Vocab: Tokenizer,
                   T_max=40,
                   L_map: Dict[str,int]=None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    X_U, Y_L = [], []
    for did, g in df.groupby("dialogue_id"):
        U_text = g["utterance"].astype(str).tolist()
        U_ids = texts_to_UxT(Vocab, U_text, T_max=T_max)   # (|U|, T)
        X_U.append(U_ids)
        if L_map is not None:
            # Prefer gold_topic when fully present; otherwise use noisy_topic
            if "gold_topic" in g.columns and g["gold_topic"].notna().all():
                labels = g["gold_topic"].astype(str).tolist()
            else:
                labels = g["noisy_topic"].astype(str).tolist()
            Y = np.array([[L_map[t]] for t in labels], dtype=np.int32)  # (|U|, 1)
            Y_L.append(Y)
        else:
            Y_L.append(None)
    return X_U, Y_L

def pad_batch(X_list: List[np.ndarray], Y_list: List[np.ndarray], pad_value=0):
    U_max = max(X.shape[0] for X in X_list)
    T = X_list[0].shape[1]
    B = len(X_list)
    X = np.full((B, U_max, T), pad_value, dtype=np.int32)
    if Y_list[0] is not None:
        Y = np.full((B, U_max, 1), 0, dtype=np.int32)
    else:
        Y = None
    for i, (Xi, Yi) in enumerate(zip(X_list, Y_list)):
        ulen = Xi.shape[0]
        X[i, :ulen, :] = Xi
        if Yi is not None:
            Y[i, :ulen, :] = Yi
    return X, Y

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation: MAE (segment counts), WindowDiff, macro P/R/F1 over L
# ─────────────────────────────────────────────────────────────────────────────
def topic_boundaries(topic_ids: List[int]) -> List[int]:
    """Boundary after utterance t if label changes: 1 else 0, for t=2..|U|."""
    b = []
    for i in range(1, len(topic_ids)):
        b.append(1 if topic_ids[i] != topic_ids[i-1] else 0)
    return b

def windowdiff(ref: List[int], hyp: List[int], k: int) -> float:
    # Pevzner & Hearst (2002)
    r = np.array(ref, dtype=np.int32)
    h = np.array(hyp, dtype=np.int32)
    U1 = len(r)
    if U1 <= 0 or k <= 1:
        return 0.0
    errors = 0
    windows = 0
    for i in range(U1 - k + 1):
        errors += (np.sum(r[i:i+k-1]) != np.sum(h[i:i+k-1]))
        windows += 1
    return errors / max(1, windows)

def macro_prf(y_true: List[int], y_pred: List[int]) -> Tuple[float,float,float]:
    labels = sorted(set(y_true))
    P, R, F = [], [], []
    for c in labels:
        tp = sum(yt==c and yp==c for yt,yp in zip(y_true,y_pred))
        fp = sum(yt!=c and yp==c for yt,yp in zip(y_true,y_pred))
        fn = sum(yt==c and yp!=c for yt,yp in zip(y_true,y_pred))
        p = tp/(tp+fp+1e-8); r = tp/(tp+fn+1e-8)
        f = 2*p*r/(p+r+1e-8)
        P.append(p); R.append(r); F.append(f)
    return float(np.mean(P)), float(np.mean(R)), float(np.mean(F))

def evaluate(model: models.Model,
             X_list: List[np.ndarray],
             Y_list: List[np.ndarray],
             L_map: Dict[str,int]) -> Dict[str,float]:
    # Window size k = 3/4 avg dialogue length (as per thesis)
    avg_len = np.mean([X.shape[0] for X in X_list])
    k = max(2, int(round(0.75 * avg_len)))

    MAEs, WDs, Ps, Rs, Fs = [], [], [], [], []
    for X_U, Y_L in zip(X_list, Y_list):
        X_pad, _ = pad_batch([X_U], [Y_L])
        Φ_logits = model.predict({"U_tokens": X_pad}, verbose=0)[0]  # (|U|, |L|)
        y_hat = np.argmax(Φ_logits, axis=-1).tolist()
        y_ref = Y_L.squeeze(-1).tolist()

        # MAE on segment counts (count boundaries per dialogue)
        b_ref = topic_boundaries(y_ref)
        b_hat = topic_boundaries(y_hat)
        MAEs.append(abs(sum(b_ref) - sum(b_hat)))

        # WindowDiff
        WDs.append(windowdiff(b_ref, b_hat, k=k))

        # Macro P/R/F1 on per-utterance labels
        p, r, f = macro_prf(y_ref, y_hat)
        Ps.append(p); Rs.append(r); Fs.append(f)

    return {
        "MAE": float(np.mean(MAEs)),
        "WindowDiff": float(np.mean(WDs)),
        "Precision": float(np.mean(Ps)),
        "Recall": float(np.mean(Rs)),
        "F1": float(np.mean(Fs)),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────
def main(args):
    # Load data and 𝔅
    DF = read_dialogue_csv(args.csv)
    BoT = read_bot_json(args.bot)

    # Build vocabulary V
    Vocab = build_vocab(DF["utterance"].astype(str).tolist(),
                        max_words=args.max_words, lower=True)

    # Embeddings E
    E = load_embeddings(args.embeddings, Vocab.word_index, d=args.emb_dim, max_words=args.max_words)
    V = min(args.max_words, len(Vocab.word_index)+1)  # vocabulary size

    # Noisy labels from 𝔅
    DF = noisy_label(DF, BoT, Vocab, E)

    # Label set L and mapping
    L = sorted(BoT.keys())
    L_map = {t:i for i, t in enumerate(L)}            # t ∈ L ↦ id
    L_inv = {i:t for t,i in L_map.items()}

    # Optional: normalise gold_topic names to L if present and not matching
    if "gold_topic" in DF.columns:
        canon = set(L)
        if not set(DF["gold_topic"].dropna().unique()).issubset(canon):
            def norm(x):
                if x in canon: return x
                for t in L:
                    if t.lower() in str(x).lower():
                        return t
                return np.nan
            DF["gold_topic"] = DF["gold_topic"].apply(norm)

    # Split by dialogue
    DF_tr, DF_dv, DF_te = split_by_dialogue(DF)

    # Pack to (U×T) and labels in L
    X_tr, Y_tr = pack_dialogues(DF_tr, Vocab, T_max=args.T_max, L_map=L_map)
    X_dv, Y_dv = pack_dialogues(DF_dv, Vocab, T_max=args.T_max, L_map=L_map)
    X_te, Y_te = pack_dialogues(DF_te, Vocab, T_max=args.T_max, L_map=L_map)

    # Build model (Θ, Φ)
    model = build_segmenter(V=V,
                            d=args.emb_dim,
                            E=E,
                            T_max=args.T_max,
                            L=len(L),
                            H_ctx=args.H_ctx)
    model.summary()

    # Mini-batch generator for variable-length dialogues
    def batcher(XL, YL, B):
        idx = np.arange(len(XL))
        while True:
            np.random.shuffle(idx)
            for i in range(0, len(idx), B):
                j = idx[i:i+B]
                Xb = [XL[k] for k in j]; Yb = [YL[k] for k in j]
                X_pad, Y_pad = pad_batch(Xb, Yb)
                yield {"U_tokens": X_pad}, Y_pad

    steps_tr = max(1, len(X_tr)//args.batch_size)
    steps_dv = max(1, len(X_dv)//args.batch_size)

    es = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    ck = callbacks.ModelCheckpoint("topic_segmenter.best.h5", save_best_only=True, monitor="val_loss")

    model.fit(
        batcher(X_tr, Y_tr, args.batch_size),
        validation_data=batcher(X_dv, Y_dv, args.batch_size),
        steps_per_epoch=steps_tr,
        validation_steps=steps_dv,
        epochs=args.epochs,
        callbacks=[es, ck],
        verbose=1
    )

    # Evaluate on test
    metrics = evaluate(model, X_te, Y_te, L_map)
    print("\n=== Test Metrics (MAE / WD / P / R / F1) ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Save per-utterance predictions
    rows = []
    for (did, g), X_U, Y_L in zip(DF_te.groupby("dialogue_id"), X_te, Y_te):
        X_pad, _ = pad_batch([X_U], [Y_L])
        Φ_logits = model.predict({"U_tokens": X_pad}, verbose=0)[0]  # (|U|, |L|)
        y_hat = np.argmax(Φ_logits, axis=-1).tolist()
        for (_, r), y in zip(g.iterrows(), y_hat):
            rows.append({
                "dialogue_id": did,
                "turn_id": r["turn_id"],
                "utterance": r["utterance"],
                "pred_topic": L_inv[y],
                "noisy_topic": r["noisy_topic"],
                "gold_topic": r.get("gold_topic", np.nan)
            })
    pd.DataFrame(rows).to_csv("test_predictions.csv", index=False)
    print("Saved predictions to test_predictions.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to dialogue CSV")
    parser.add_argument("--bot", type=str, required=True, help="Path to Bag-of-Topics JSON")
    parser.add_argument("--embeddings", type=str, default="", help="Path to GloVe .txt (optional)")
    parser.add_argument("--max_words", type=int, default=50000, help="|V| cap")
    parser.add_argument("--emb_dim", type=int, default=300, help="d (embedding dimension)")
    parser.add_argument("--T_max", type=int, default=40, help="T (max tokens per utterance)")
    parser.add_argument("--H_ctx", type=int, default=256, help="context BiLSTM hidden per direction")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    main(args)
