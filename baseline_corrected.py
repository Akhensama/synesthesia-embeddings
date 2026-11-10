"""
Baseline-corrected comparison of embedding similarity vs. synesthetic frequency.

The original analysis (results_summary.ipynb) correlated the raw 26 x 11 matrices.
Both matrices carry baselines that say nothing about *which letter* goes with
*which color*:

  - color baseline: synesthetes pick some colors more often for every letter
    (blue, orange, yellow), and some color words sit closer to every letter in
    embedding space.
  - letter baseline: some letter embeddings are closer to all colors than others.

Matching color baselines alone produces a positive raw Spearman rho. Here we

  1. double-center both matrices (remove letter and color means) before Spearman,
  2. test significance by shuffling letters (rows) of the embedding matrix, which
     keeps every baseline intact and only breaks the letter-color pairing,
  3. score hits on color-centered similarities ("which color is unusually close to
     this letter?") and compare them to the same letter-shuffle null,
  4. Holm-correct the permutation p-values across all conditions.

Run:  python baseline_corrected.py   -> prints a table, writes data/corrected_results.json
"""
import json
from pathlib import Path

import numpy as np
import scipy.io
from scipy import stats
import gensim.downloader as gensim_api
from sentence_transformers import SentenceTransformer

DATA_DIR    = Path('data')
LETTERS     = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
COLOR_NAMES = ['Black','White','Red','Green','Yellow','Blue','Brown','Purple','Pink','Orange','Gray']
COLOR_WORDS = [c.lower() for c in COLOR_NAMES]
N_PERM      = 10_000
rng         = np.random.default_rng(0)

# ── Synesthesia frequency matrix (same as results_summary.ipynb) ─────────────
mat        = scipy.io.loadmat(DATA_DIR / 'EaglemanColoredAlphabets.mat')
lrgb_mat   = scipy.io.loadmat(DATA_DIR / 'lRGBnathan.mat')
rgb_data   = mat['u_rgb'][:, 1:4, :]
label_grid = lrgb_mat['labeledRGB']
_r = np.linspace(0, 255, label_grid.shape[0])
_g = np.linspace(0, 255, label_grid.shape[1])
_b = np.linspace(0, 255, label_grid.shape[2])

def classify_rgb_batch(rgb_arr):
    labels = np.full(len(rgb_arr), -1, dtype=np.int8)
    for i, (r, g, b) in enumerate(rgb_arr):
        if np.isnan(r): continue
        ri = int(np.argmin(np.abs(_r - r * 255)))
        gi = int(np.argmin(np.abs(_g - g * 255)))
        bi = int(np.argmin(np.abs(_b - b * 255)))
        labels[i] = label_grid[ri, gi, bi]
    return labels

freq = np.zeros((26, 11))
for li in range(26):
    labels = classify_rgb_batch(rgb_data[:, :, li])
    valid  = labels[labels >= 0]
    for ci in range(11):
        freq[li, ci] = np.sum(valid == ci) / len(valid)
modal = freq.argmax(1)

# ── Similarity matrices ──────────────────────────────────────────────────────
def cosim(L, C):
    L = np.array(L, dtype=float); C = np.array(C, dtype=float)
    L /= np.linalg.norm(L, axis=1, keepdims=True)
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    return L @ C.T

def load_gpt(path):
    c = json.loads((DATA_DIR / path).read_text())
    return cosim(c['letter_embs'], c['color_embs'])

w2v = gensim_api.load('word2vec-google-news-300')
glove = gensim_api.load('glove-twitter-200')
st = SentenceTransformer('all-roberta-large-v1')
def st_sim(lq, cq):
    return st.encode(lq, normalize_embeddings=True) @ st.encode(cq, normalize_embeddings=True).T
voy = json.loads((DATA_DIR / 'voyage3_embeddings_cache.json').read_text())

CONDITIONS = {
    'RoBERTa-large — "A"':             st_sim(LETTERS, COLOR_WORDS),
    'RoBERTa-large — "letter A"':      st_sim([f'letter {l}' for l in LETTERS], COLOR_WORDS),
    'RoBERTa-large — "the letter A"':  st_sim([f'the letter {l}' for l in LETTERS],
                                              [f'the color {c}' for c in COLOR_WORDS]),
    'Voyage-3 — "A"':                  cosim(voy['bare_letters'], voy['colors']),
    'Voyage-3 — "letter A"':           cosim(voy['letter_letters'], voy['colors']),
    'Voyage-3 — "the letter A"':       cosim(voy['the_letters'], voy['the_colors']),
    'GPT — "A" / "red"':               load_gpt('gpt_embeddings_cache_bare.json'),
    'GPT — "letter A" / "red"':        load_gpt('gpt_embeddings_cache_v2.json'),
    'GPT — "the letter A" / "the color"': load_gpt('gpt_embeddings_cache.json'),
    'GloVe Twitter — "a"':             cosim([glove[l.lower()] for l in LETTERS],
                                             [glove[c] for c in COLOR_WORDS]),
    'W2V Google News — "A"':           cosim([w2v[l] for l in LETTERS], [w2v[c] for c in COLOR_WORDS]),
    'W2V Google News — "letter A"':    cosim([(w2v['letter'] + w2v[l]) / 2 for l in LETTERS],
                                             [w2v[c] for c in COLOR_WORDS]),
    'W2V Google News — "the letter A"': cosim([(w2v['the'] + w2v['letter'] + w2v[l]) / 3 for l in LETTERS],
                                              [(w2v['the'] + w2v['color'] + w2v[c]) / 3 for c in COLOR_WORDS]),
}

# ── Baseline-corrected statistics ────────────────────────────────────────────
def double_center(M):
    return M - M.mean(0, keepdims=True) - M.mean(1, keepdims=True) + M.mean()

def rho_c(S):
    return stats.spearmanr(double_center(freq).ravel(), double_center(S).ravel())[0]

def hits_c(S):
    return int((((S - S.mean(0)).argmax(1)) == modal).sum())

perms = [rng.permutation(26) for _ in range(N_PERM)]
boots = [rng.integers(0, 26, 26) for _ in range(N_PERM)]

def rho_c_rows(S, idx):
    return stats.spearmanr(double_center(freq[idx]).ravel(), double_center(S[idx]).ravel())[0]

rows = []
for label, S in CONDITIONS.items():
    rho_raw, p_raw = stats.spearmanr(freq.ravel(), S.ravel())
    r, h = rho_c(S), hits_c(S)
    null_r = np.array([rho_c(S[p]) for p in perms])
    null_h = np.array([hits_c(S[p]) for p in perms])
    # 95% CI from resampling letters (the 26 letters, not the 286 cells, are the units)
    boot_r = np.array([rho_c_rows(S, b) for b in boots])
    rows.append({
        'ci': [round(float(x), 3) for x in np.percentile(boot_r, [2.5, 97.5])],
        'label': label,
        'rho_raw': round(float(rho_raw), 4), 'p_raw': float(p_raw),
        'hits_raw': int((S.argmax(1) == modal).sum()),
        'rho': round(float(r), 4),
        'p': float((np.sum(null_r >= r) + 1) / (N_PERM + 1)),
        'hits': h,
        'hits_null': round(float(null_h.mean()), 2),
        'p_hits': float((np.sum(null_h >= h) + 1) / (N_PERM + 1)),
        'hit_letters': [LETTERS[i] for i in range(26) if (S[i] - S.mean(0)).argmax() == modal[i]],
    })

# Holm correction across conditions
for key in ('p', 'p_hits'):
    order = np.argsort([r[key] for r in rows])
    m, running = len(rows), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * rows[i][key]))
        rows[i][key + '_holm'] = running

def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'

for r in rows:
    r['sig'] = stars(r['p_holm'])

print(f'{"Condition":<38} {"raw ρ":>7} {"ρ (corr.)":>9} {"p perm":>7} {"Holm":>6}  '
      f'{"hits raw":>8} {"hits":>5} {"null":>5} {"p hits":>7}')
for r in rows:
    print(f'{r["label"]:<38} {r["rho_raw"]:>+7.3f} {r["rho"]:>+9.3f} {r["p"]:>7.4f} {r["sig"]:>6}  '
          f'{r["hits_raw"]:>8} {r["hits"]:>5} {r["hits_null"]:>5.1f} {r["p_hits"]:>7.4f}')
print(f'\nAlways guessing the single most common modal color: {np.bincount(modal).max()}/26')
print('\n95% bootstrap CIs (resampling letters):')
for r in rows:
    print(f'  {r["label"]:<38} {r["rho"]:+.3f}  [{r["ci"][0]:+.3f}, {r["ci"][1]:+.3f}]')

# Phrasing effect: is full phrasing better than bare letters within each model?
# Paired bootstrap over letters on the difference in corrected rho.
print('\nPhrasing effect, corrected rho("the letter A") − rho("A"), paired bootstrap over letters:')
phrasing = []
for model in ('RoBERTa-large', 'Voyage-3', 'GPT'):
    bare = next(S for k, S in CONDITIONS.items() if k.startswith(model) and '"A"' in k)
    full = next(S for k, S in CONDITIONS.items() if k.startswith(model) and 'the letter A' in k)
    diff = rho_c(full) - rho_c(bare)
    boot_d = np.array([rho_c_rows(full, b) - rho_c_rows(bare, b) for b in boots])
    lo, hi = np.percentile(boot_d, [2.5, 97.5])
    p_one = float((np.sum(boot_d <= 0) + 1) / (N_PERM + 1))
    phrasing.append({'model': model, 'diff': round(float(diff), 3),
                     'ci': [round(float(lo), 3), round(float(hi), 3)], 'p': p_one})
    print(f'  {model:<14} Δρ = {diff:+.3f}  [{lo:+.3f}, {hi:+.3f}]  one-sided p = {p_one:.4f}')

(DATA_DIR / 'corrected_results.json').write_text(json.dumps(
    {'n_perm': N_PERM, 'always_guess_best': int(np.bincount(modal).max()),
     'results': rows, 'phrasing_effect': phrasing},
    indent=1, ensure_ascii=False))
