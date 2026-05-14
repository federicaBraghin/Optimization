"""
dataset.py
----------
Generazione dei dataset sintetici per gli esperimenti (Gruppo 66, A.Y. 2025/26).

Tre configurazioni in R^2 con k=3 cluster Gaussiani:
  - easy   : cluster bilanciati, ben separati      → vincolo capacità NON morde
  - medium : cluster sbilanciati, parziale overlap  → vincolo capacità morde
  - hard   : fortemente sbilanciati, molto overlap  → vincolo capacità morde forte

Usare generate_all() per ottenere tutti e tre i dataset.
Usare generate_experiment() per istanze con parametri liberi.
Usare plot_datasets() per visualizzarli.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Centri dei cluster in R^2 (usati solo per easy/medium/hard)
# Spaziati per rendere il caso easy davvero separato
# ---------------------------------------------------------------------------

CENTERS_R2 = np.array([
    [0.0, 0.0],   # cluster 0
    [6.0, 0.0],   # cluster 1
    [3.0, 5.0],   # cluster 2
], dtype=float)


# ---------------------------------------------------------------------------
# Configurazioni
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    """
    Configurazione di un dataset sintetico.

    Attributi
    ---------
    name          : nome ('easy', 'medium', 'hard', 'experiment')
    n_per_cluster : numero di punti per ogni cluster vero
    std           : deviazione standard delle Gaussiane
    random_state  : seed per riproducibilità
    """
    name: str
    n_per_cluster: list
    std: float
    random_state: int = 42


CONFIGS = {
    # Cluster bilanciati (30/30/30), ben separati (std bassa).
    # m = ceil(90/3) = 30, dimensione max = 30 → vincolo NON morde.
    # A1 dovrebbe sempre trovare l'ottimo.
    "easy": DatasetConfig(
        name="easy",
        n_per_cluster=[30, 30, 30],
        std=0.8,
    ),

    # Cluster sbilanciati (40/30/20), parziale sovrapposizione.
    # m = ceil(90/3) = 30, dimensione max = 40 → vincolo morde (10 punti ridistribuiti).
    # A1 deve lavorare di più per trovare una buona assegnazione.
    "medium": DatasetConfig(
        name="medium",
        n_per_cluster=[40, 30, 20],
        std=1.4,
    ),

    # Cluster fortemente sbilanciati (50/25/15), molto overlap.
    # m = ceil(90/3) = 30, dimensione max = 50 → vincolo morde forte (20 punti ridistribuiti).
    # La ridistribuzione forzata rende difficile trovare l'ottimo per A1.
    "hard": DatasetConfig(
        name="hard",
        n_per_cluster=[50, 25, 15],
        std=2.0,
    ),
}


# ---------------------------------------------------------------------------
# Struttura dati risultante
# ---------------------------------------------------------------------------

@dataclass
class Dataset:
    """
    Dataset generato con metadati utili per gli esperimenti.

    Attributi
    ---------
    X        : array (n, d) — punti del dataset
    y        : array (n,)   — etichette vere (NON usate nel clustering)
    k        : numero di cluster
    m        : capacità massima per centroide (>= ceil(n/k))
    config   : DatasetConfig usata per generare questo dataset
    """
    X: np.ndarray
    y: np.ndarray
    k: int
    m: int
    config: DatasetConfig

    @property
    def n(self):
        return len(self.X)

    @property
    def d(self):
        return self.X.shape[1]

    @property
    def constraint_bites(self):
        """True se il vincolo di capacità forza una redistribuzione."""
        return max(self.config.n_per_cluster) > self.m


# ---------------------------------------------------------------------------
# Generazione — dataset fissi (easy / medium / hard)
# ---------------------------------------------------------------------------

def generate_dataset(config: DatasetConfig, centers=CENTERS_R2) -> Dataset:
    """
    Genera un Dataset da una DatasetConfig usando Gaussian blobs.

    Parametri
    ---------
    config  : DatasetConfig con n_per_cluster, std, random_state
    centers : array (k, d) con i centri dei cluster

    Restituisce
    -----------
    Dataset con X, y, k, m calcolati dai dati
    """
    k = len(config.n_per_cluster)
    n = sum(config.n_per_cluster)

    X, y = make_blobs(
        n_samples=config.n_per_cluster,
        centers=centers,
        cluster_std=config.std,
        random_state=config.random_state,
    )

    m = math.ceil(n / k)

    return Dataset(X=X, y=y, k=k, m=m, config=config)


# ---------------------------------------------------------------------------
# Generazione — istanze parametriche per gli esperimenti
# ---------------------------------------------------------------------------

def generate_experiment(n: int, k: int, d: int = 2,
                        m: int = None,
                        std: float = 1.0,
                        random_state: int = None) -> Dataset:
    """
    Genera un dataset casuale per gli esperimenti con parametri liberi.

    I punti xi appartengono a un iper-rettangolo Q = prod_l [L^l, U^l]
    determinato dai dati stessi (L^l = min_i x_i^l, U^l = max_i x_i^l),
    coerentemente con la Sezione 1.1 del report.

    Parametri
    ---------
    n            : numero totale di punti
    k            : numero di cluster
    d            : dimensionalità (default 2)
    m            : capacità per centroide (default ceil(n/k))
    std          : deviazione standard delle Gaussiane
    random_state : seed (None = casuale, intero = riproducibile)

    Restituisce
    -----------
    Dataset con X, y, k, m e config
    """
    rng = np.random.default_rng(random_state)

    # Centri campionati casualmente in R^d.
    # I bound L^l = min_i x_i^l e U^l = max_i x_i^l
    # dell'iper-rettangolo Q sono determinati dai punti generati,
    # coerentemente con la Sezione 1.1 del report.
    centers = rng.standard_normal(size=(k, d))

    # Distribuzione bilanciata: base punti per cluster, resto distribuito
    base = n // k
    remainder = n % k
    n_per_cluster = [base + 1 if i < remainder else base for i in range(k)]

    X, y = make_blobs(
        n_samples=n_per_cluster,
        centers=centers,
        cluster_std=std,
        random_state=random_state,
    )

    if m is None:
        m = math.ceil(n / k)

    cfg = DatasetConfig(
        name="experiment",
        n_per_cluster=n_per_cluster,
        std=std,
        random_state=random_state if random_state is not None else -1,
    )

    return Dataset(X=X, y=y, k=k, m=m, config=cfg)


# ---------------------------------------------------------------------------
# Generazione — tutti i dataset fissi
# ---------------------------------------------------------------------------

def generate_all(centers=CENTERS_R2) -> dict:
    """
    Genera tutti e tre i dataset fissi (easy, medium, hard).

    Restituisce
    -----------
    dict con chiavi 'easy', 'medium', 'hard' e valori Dataset
    """
    return {name: generate_dataset(cfg, centers)
            for name, cfg in CONFIGS.items()}


# ---------------------------------------------------------------------------
# Visualizzazione
# ---------------------------------------------------------------------------

def plot_datasets(datasets: dict, save_path=None):
    """
    Visualizza i tre dataset in R^2 con i cluster veri colorati.

    Parametri
    ---------
    datasets  : dict restituito da generate_all()
    save_path : se fornito, salva la figura in quel percorso
    """
    COLORS  = ["#378ADD", "#1D9E75", "#D85A30"]
    MARKERS = ["o", "s", "^"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for ax, name in zip(axes, ["easy", "medium", "hard"]):
        ds = datasets[name]

        for j in range(ds.k):
            mask = ds.y == j
            ax.scatter(
                ds.X[mask, 0], ds.X[mask, 1],
                c=COLORS[j], marker=MARKERS[j],
                s=30, alpha=0.75, linewidths=0,
                label=f"Cluster {j}  (n={ds.config.n_per_cluster[j]})",
            )

        bites_str = "SÌ" if ds.constraint_bites else "NO"
        ax.set_title(
            f"{name.capitalize()}  —  n={ds.n}, k={ds.k}, m={ds.m}\n"
            f"std={ds.config.std}  |  vincolo morde: {bites_str}",
            fontsize=10,
        )
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.35)

    fig.suptitle("Dataset sintetici (R²) — cluster veri colorati", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figura salvata in: {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# Riepilogo testuale
# ---------------------------------------------------------------------------

def print_summary(datasets: dict):
    """Stampa una tabella riassuntiva dei dataset."""
    print(f"\n{'Config':<10} {'n':>4} {'k':>3} {'m':>4}  "
          f"{'sizes':<14} {'std':>5}  {'vincolo morde':>14}")
    print("-" * 62)
    for name in ["easy", "medium", "hard"]:
        ds = datasets[name]
        sizes = " / ".join(str(s) for s in ds.config.n_per_cluster)
        bites = "SÌ" if ds.constraint_bites else "NO"
        print(f"{name:<10} {ds.n:>4} {ds.k:>3} {ds.m:>4}  "
              f"{sizes:<14} {ds.config.std:>5}  {bites:>14}")
    print()