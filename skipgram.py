import numpy as np


def sigmoid(x):
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ez = np.exp(x[neg])
    out[neg] = ez / (1.0 + ez)
    return out
    

# TODO 1: Implementa un método de entrenamiento simple, esto es, con learning rate (LR) constante y ventana estática.
class Trainer:
    def _neg_sampling_fix(self):
        # TODO 2: Inicializa `self.neg_prob`, que será usado como distribución de probabilidad a la hora de hacer el muestreo negativo, de modo que contenga las frecuencias absolutas de cada token del vocabulario elevadas a 3/4 y normalizadas, de modo que las probabilidades resultantes sumen 1.
        if not hasattr(self, "token_freq"):
            raise ValueError("token frequencies are not initialized")

        freq_pow = np.power(self.token_freq, 0.75)
        total = freq_pow.sum()
        if total == 0:
            self.neg_prob = np.full(self.vocab_size, 1.0 / self.vocab_size, dtype=np.float64)
        else:
            self.neg_prob = freq_pow / total

    def _subsample_data(self):
        # TODO 3: Reduce la ocurrencia de los tokens más frecuentes usando la siguiente fórmula:
        # `p_keep = (np.sqrt(t / f) + t / f) if f > 0 else 1.0`
        # donde `t = 1e-5` y `f` es la frecuencia relativa del token.
        if self.corpus_ids.size == 0:
            return

        t = 1e-5
        rel_freq = np.zeros(self.vocab_size, dtype=np.float64)
        if self.total_tokens > 0:
            rel_freq = self.token_freq / float(self.total_tokens)

        keep_prob = np.ones(self.vocab_size, dtype=np.float64)
        mask = rel_freq > 0
        freq_vals = rel_freq[mask]
        keep_prob[mask] = np.clip(np.sqrt(t / freq_vals) + t / freq_vals, 0.0, 1.0)

        rand_vals = self.rng.random(self.corpus_ids.shape[0])
        selected = rand_vals < keep_prob[self.corpus_ids]
        self.corpus_ids = self.corpus_ids[selected]
        self.total_tokens = int(self.corpus_ids.size)
        self.token_freq = np.bincount(self.corpus_ids, minlength=self.vocab_size).astype(np.float64)

    def __init__(self, corpus_fpath, rng, embedding_dim, window_size, epochs, lr, lr_min_factor, neg_samples):
        self.corpus_fpath = corpus_fpath
        self.rng = rng
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.epochs = epochs
        self.lr = lr
        self.lr_min_factor = lr_min_factor 
        self.neg_samples = neg_samples

        # TODO 1.1: Carga el corpus y tokenízalo usando el tokenizador BPE de la práctica anterior.
        # El corpus debería quedar codificado como una secuencia de ids de tokens.
        import json
        from pathlib import Path
        from BPE.bpe import ByteLevelBPE

        model_path = Path(__file__).resolve().parent / "BPE" / "modelo.json"
        with open(model_path, "r", encoding="utf-8") as f:
            model_data = json.load(f)

        bpe = ByteLevelBPE()
        bpe.merges = [
            (tuple(pair[0]), tuple(pair[1]))
            for pair in model_data["merges"]
        ]
        bpe.id2bytes = [tuple(token) for token in model_data["id2bytes"]]
        bpe.vocab = {token: idx for idx, token in enumerate(bpe.id2bytes)}
        self.bpe = bpe

        corpus_path = Path(self.corpus_fpath)
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus_text = f.read()

        corpus_ids = bpe.encode(corpus_text)
        self.corpus_ids = np.array(corpus_ids, dtype=np.int32)
        self.vocab_size = len(bpe.id2bytes)
        if self.vocab_size == 0:
            raise ValueError("El vocabulario del tokenizador está vacío")

        if self.corpus_ids.size == 0:
            self.token_freq = np.zeros(self.vocab_size, dtype=np.float64)
        else:
            self.token_freq = np.bincount(self.corpus_ids, minlength=self.vocab_size).astype(np.float64)
        self.total_tokens = int(self.corpus_ids.size)

        # Aplica ajustes para evitar la sobreponderancia de tokens frecuentes
        self._neg_sampling_fix()
        self._subsample_data()

    def sample_neg(self, forbidden):
        # TODO 1.2: Obtén una muestra negativa de tokens, evitando seleccionar aquellos en `forbidden`, que serán los que estén dentro de la ventana actual.
        forbidden_set = set(int(tok) for tok in forbidden)
        prob = np.array(self.neg_prob, copy=True)
        idxs = np.array([], dtype=np.int64)

        if forbidden_set:
            idxs = np.fromiter(forbidden_set, dtype=np.int64, count=len(forbidden_set))
            idxs = idxs[(idxs >= 0) & (idxs < self.vocab_size)]
            prob[idxs] = 0.0

        prob_sum = prob.sum()
        if prob_sum <= 0:
            allowed = np.ones(self.vocab_size, dtype=bool)
            if forbidden_set:
                allowed[idxs] = False
            candidates = np.flatnonzero(allowed)
            if candidates.size == 0:
                raise ValueError("No hay tokens disponibles para muestreo negativo")
            return self.rng.choice(candidates, size=self.neg_samples, replace=True)

        prob /= prob_sum
        return self.rng.choice(self.vocab_size, size=self.neg_samples, replace=True, p=prob)

    def train(self):
        # TODO 1.3: Inicializa dos matrices de `self.vocab_size` x `self.embedding_dim` para tokens centrales y contexto.
        center_embeddings = self.rng.normal(
            0.0, 0.1, size=(self.vocab_size, self.embedding_dim)
        ).astype(np.float32)
        context_embeddings = self.rng.normal(
            0.0, 0.1, size=(self.vocab_size, self.embedding_dim)
        ).astype(np.float32)

        # TODO 1.4: Para cada `epoch` y para cada token en el corpus:
        # Para cada token en el contexto del token actual, es decir, para cada token dentro de los `self.window_size` tokens a la derecha e izquieda del actual, sin contar este:
        # Calcular el producto escalar entre las embeddings del token central y token de contexto.
        # Pasar el resultado por la función `sigmoid`, obteniendo `pos_score`.
        # Muestra positiva: actualizar las embeddings del token central y token contexto usando el LR, `(1 - pos_score)` y la embedding (¡original!) del otro token.
        # Muestras negativas: obtener muestras negativas para el token central y, para cada una, realizar un proceso similar al de la muestra positiva, con la salvedad de que ahora `pos_score` es `neg_score` y se usa `-neg_score` para actualizar las embeddings.

        # TODO 4: Usa una ventana de contexto dinámica, con tamaños que varíen aleatoriamente dentro del rango de la ventana estática original.
        # TODO 5: Haz que el LR disminuya progresivamente durante el entrenamiento (linear decay).

        # TODO 1.5: Devuelve las dos matrices de embeddings.
        if self.corpus_ids.size == 0 or self.window_size <= 0:
            return center_embeddings, context_embeddings

        min_lr = self.lr * self.lr_min_factor
        total_steps = max(1, self.epochs * self.corpus_ids.size)
        step = 0

        corpus_len = self.corpus_ids.size
        for _ in range(self.epochs):
            for idx in range(corpus_len):
                center_token = int(self.corpus_ids[idx])
                step += 1
                decay = step / total_steps
                current_lr = max(min_lr, self.lr - (self.lr - min_lr) * decay)

                dynamic_window = int(self.rng.integers(1, self.window_size + 1))
                left = max(0, idx - dynamic_window)
                right = min(corpus_len, idx + dynamic_window + 1)

                if right - left <= 1:
                    continue

                window_tokens = set(int(tok) for tok in self.corpus_ids[left:right])
                context_positions = list(range(left, idx)) + list(range(idx + 1, right))
                for ctx_idx in context_positions:
                    context_token = int(self.corpus_ids[ctx_idx])

                    center_vec = center_embeddings[center_token].copy()
                    context_vec = context_embeddings[context_token].copy()
                    dot = float(np.dot(center_vec, context_vec))
                    pos_score = sigmoid(np.array([dot], dtype=np.float32))[0]
                    grad = current_lr * (1.0 - pos_score)
                    center_embeddings[center_token] += grad * context_vec
                    context_embeddings[context_token] += grad * center_vec

                    neg_samples = self.sample_neg(window_tokens)
                    for neg_token in neg_samples:
                        neg_token = int(neg_token)
                        center_before_neg = center_embeddings[center_token].copy()
                        neg_vec = context_embeddings[neg_token].copy()
                        dot_neg = float(np.dot(center_before_neg, neg_vec))
                        neg_score = sigmoid(np.array([dot_neg], dtype=np.float32))[0]
                        grad_neg = current_lr * (-neg_score)
                        center_embeddings[center_token] += grad_neg * neg_vec
                        context_embeddings[neg_token] += grad_neg * center_before_neg

        return center_embeddings, context_embeddings


def dump_embeddings(
        # ...
        E
        ):
    # TODO 1.6: Escribe las embeddings en un fichero de texto donde, en la primera fila, aparezca el tamaño del vocabulario y el número de dimensiones de las embeddings y, en el resto de filas, cada token seguido de su correspondiente embedding, separando cada elemento con espacios simples. Ojo, los tokens pueden contener espacios.
    import json
    from pathlib import Path

    model_path = Path(__file__).resolve().parent / "BPE" / "modelo.json"
    with open(model_path, "r", encoding="utf-8") as f:
        model_data = json.load(f)

    id2bytes = [tuple(token) for token in model_data["id2bytes"]]
    vocab_size = min(len(id2bytes), E.shape[0])
    output_path = Path(__file__).resolve().parent / "embeddings.txt"

    with open(output_path, "w", encoding="utf-8") as out:
        out.write(f"{vocab_size} {E.shape[1]}\n")
        for idx in range(vocab_size):
            token_bytes = id2bytes[idx]
            try:
                token_text = bytes(token_bytes).decode("utf-8")
            except UnicodeDecodeError:
                token_text = ""

            if not token_text or any(ch.isspace() for ch in token_text):
                hex_repr = " ".join(f"{b:02x}" for b in token_bytes)
                token_text = f"[{hex_repr}]"

            vec = " ".join(f"{value:.6f}" for value in E[idx].tolist())
            out.write(f"{token_text} {vec}\n")


def main():
    trainer = Trainer(
        corpus_fpath='./tiny_cc_news.txt',
        rng=np.random.default_rng(42),
        embedding_dim=100,
        window_size=5,
        epochs=5,
        lr=0.05,
        lr_min_factor=0.0001,
        neg_samples=5,
    )

    T, C = trainer.train()
    E = (T + C) / 2.0  # Matriz final de embeddings
    dump_embeddings(
        # ...
        E
        )
    

if __name__ == "__main__":
    main()
