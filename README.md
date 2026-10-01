# Cooperation: open-mindedness in multidimensional opinion dynamics

Simulation code for the open-mindedness extension of the Sîrbu–Loreto
multidimensional opinion model (Sîrbu et al. 2013), plus one exploratory
analysis of Italian political retweet networks.

Authors: Alice Nappa, Camilla Gatti, Tommaso Pisani.

## Setup

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt        # runtime
.venv/bin/pip install -r requirements-dev.txt    # + headless notebook execution
```

Tested with Python 3.11.15 and the versions pinned in `requirements.txt`.

## Contents

| File | What it is |
|---|---|
| `Functions_sirbu_loreto.py` | Baseline Sîrbu–Loreto model: overlap, update rule, population generation, external sources, clustering. |
| `Functions_OPEN_MINDEDNESS.py` | Open-minded variant. Does `from Functions_sirbu_loreto import *` and then redefines several functions (see below). |
| `Functions_explained.ipynb` | Walk-through of both modules (Italian/English). |
| `MODELLO_SIRBU_LORETO.ipynb` | Baseline model: entropy vs overlap, consensus/polarisation transition vs initial overlap. |
| `MODELLO_SIRBU_LORETO_con_OPEN_MINDEDNESS.ipynb` | Open-minded model with memory window `mu`, and with external sources `external_info(K, a)`. |
| `Cross_community_retweets.ipynb` | Louvain communities and cross-community retweet share among Italian political leaders. Needs external data (not in repo). |

### Name shadowing between the two modules

`Functions_OPEN_MINDEDNESS.py` re-exports every function from
`Functions_sirbu_loreto.py` and then redefines some of them. Importing
`Functions_OPEN_MINDEDNESS` therefore gives you:

- from the open-minded module: `calculate_overlap` (not clipped to [0, 1]),
  `generate_population` (keeps 10% of high-entropy draws, returns a list),
  `hierarchical_clustering_K`, `interact_with_info`, `calculate_w`,
  `interact_individuals_om`, `evolve_population`,
  `evolve_population_with_info_and_peer_openm`;
- from the baseline module: `interact_individuals`, `update`,
  `evolve_population_1/2`, `evolve_population_with_info`,
  `evolve_population_with_info_and_peer`.

Functions keep resolving names in the module where they are defined, so
`interact_individuals` always uses the baseline (clipped) `calculate_overlap`.

## Running notebooks headless

```bash
.venv/bin/jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=600 --output /tmp/out.ipynb Functions_explained.ipynb
```

No random seed is set anywhere, so results differ between runs.

## Data

`Cross_community_retweets.ipynb` reads `tweets_light.parquet`,
`retweets_light.parquet`, `pages_net1.csv` and `lista_leader.txt`. These are
not distributed with this repository.

## Licence

MIT, see `LICENSE`.
