# ============================================================
# AlphaQuant (ICLR-2025 style, Named & Persistent Version)
# LLM + LightGBM + Rolling Backtest + Feedback Loop
# ============================================================

!pip -q install yfinance lightgbm openai torch shap matplotlib seaborn

import os, re, json, torch, warnings, numpy as np, pandas as pd
from openai import OpenAI
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error, ndcg_score
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import shap
warnings.filterwarnings("ignore")

# ---------- Configuration ----------
API_KEY = "YOUR_API_KEY"
client = OpenAI(api_key=API_KEY)
SYMBOL = "AAPL"
ROUNDS = 10
NUM_FEATURES = 5
SAVE_DIR = "./generated_features"
os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# Data Loading and Preprocessing
# ============================================================
import yfinance as yf
data = yf.download(SYMBOL, period="5y")
data["log_ret"] = np.log(data["Close"] / data["Close"].shift(1))
data.dropna(inplace=True)
print(f"Loaded {SYMBOL}, samples = {len(data)}")

log_returns = torch.tensor(data["log_ret"].values, dtype=torch.float32)
y_future = np.roll(data["log_ret"].values, -1)[:-1]
log_returns = log_returns[:-1]


# ============================================================
# Prompt Builder
# ============================================================
def build_prompt(top_k_examples, eliminated_list, previous_errors, feedback_summary, num_features=3):
    system_prompt = f"""
You are a senior quantitative researcher.
Generate compact, correct PyTorch code for time-series feature engineering.

Rules:
- Exactly {num_features} unique functions, each with a meaningful name
  that reflects its financial/statistical intuition.
  Examples: momentum_score, rolling_volatility, mean_reversion_strength, kurtosis_signal.
  Avoid generic names like feature_1.
- Signature: def <function_name>(log_returns: torch.Tensor) -> torch.Tensor
- Return 1D tensor of SAME LENGTH as log_returns.
- Use only tensor ops; no print, imports, loops, markdown fences.
- Ensure non-constant outputs and safe tensor math.
- Avoid NaN/Inf via (x + 1e-8) style stabilization.
- Use operations like torch.cumsum, torch.std, torch.tanh, torch.sign, torch.clamp.
- Output interpretable, predictive signals inspired by finance literature (risk, momentum, volatility, etc.).
"""
    user_prompt = f"""
Seen examples:
{top_k_examples}

Performance feedback from the previous round:
{feedback_summary}

Avoid redundancy:
{eliminated_list}

Avoid previous errors:
{previous_errors}

Now generate {num_features} new, unique, interpretable PyTorch feature extraction functions.
Focus on improving features with higher Spearman correlation and lower MAE.
"""
    return {"system": system_prompt.strip(), "user": user_prompt.strip()}


# ============================================================
# Rolling Cross-Validation Evaluation
# ============================================================
def rolling_backtest_eval(feature, y_future, n_splits=5, overlap=0.5):
    N = len(feature)
    win_size = int(N / (n_splits + 1))
    step = int(win_size * (1 - overlap))
    maes, sps, ndcgs = [], [], []

    for start in range(0, N - win_size, step):
        end = start + win_size
        if end + win_size >= N: break
        X_train, y_train = feature[start:end].reshape(-1,1), y_future[start:end]
        X_valid, y_valid = feature[end:end+win_size].reshape(-1,1), y_future[end:end+win_size]

        model = LGBMRegressor(n_estimators=80, learning_rate=0.05, max_depth=3)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)

        mae = mean_absolute_error(y_valid, y_pred)
        sp = spearmanr(y_valid, y_pred)[0] or 0
        y_valid_clip = np.clip(y_valid, 0, None)
        y_pred_clip = np.clip(y_pred - np.min(y_pred), 0, None)
        nd = ndcg_score([y_valid_clip], [y_pred_clip]) if np.sum(y_pred_clip) > 0 else 0

        maes.append(mae); sps.append(sp); ndcgs.append(nd)

    MAE, SP, NDCG = np.nanmean(maes), np.nanmean(sps), np.nanmean(ndcgs)
    Score = 0.5*(1 - MAE) + 0.3*SP + 0.2*NDCG
    return MAE, SP, NDCG, Score


def evaluate_feature(feat, y_future):
    if len(feat) != len(y_future): return np.nan, np.nan, np.nan, np.nan
    feat = np.nan_to_num(feat)
    y_future = np.nan_to_num(y_future)
    return rolling_backtest_eval(feat, y_future, n_splits=5, overlap=0.5)


def compute_test_sharpe(X, y, test_ratio=0.1):
    split = int((1 - test_ratio) * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    model = LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=4)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    signal = np.tanh(y_pred - np.mean(y_pred))
    ret = signal * y_test
    return np.mean(ret) / (np.std(ret) + 1e-9)


# ============================================================
# LLM Feature Generator and Saving Logic
# ============================================================
def llm_generate_features(client, model_name, log_returns_tensor, top_examples, eliminated, prev_errors, feedback, num_features=3):
    prompt = build_prompt(top_examples, eliminated, prev_errors, feedback, num_features)
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]}
        ],
        temperature=0.8,
    )
    raw_code = re.sub(r"```(python)?", "", resp.choices[0].message.content).strip()
    print("[DEBUG] LLM output (first 300 chars):\n", raw_code[:300])

    local_env = {"torch": torch, "np": np}
    features, func_names, errors = [], [], []
    try:
        exec(raw_code, local_env)
        for name, obj in local_env.items():
            if callable(obj) and name not in ["torch", "np"]:
                try:
                    val = obj(log_returns_tensor)
                    if isinstance(val, torch.Tensor):
                        arr = np.resize(val.detach().cpu().numpy().flatten(), len(log_returns_tensor))
                        if np.isfinite(arr).all():
                            features.append(arr)
                            func_names.append(name)
                            print(f"[FUNC] {name}: valid, len={len(arr)}")
                except Exception as e:
                    errors.append(f"{name}: {str(e)}")
    except Exception as e:
        errors.append(str(e))

    return features, func_names, errors, raw_code


# ============================================================
# Main Training Loop
# ============================================================
history, mae_hist, sp_hist, ndcg_hist, importance_hist = [], [], [], [], []
top_examples, eliminated, prev_errors, feedback = "", "", "", ""
feats_global = []

for rnd in range(1, ROUNDS + 1):
    print(f"\n===== Round {rnd} =====")
    feats, names, errors, code_str = llm_generate_features(
        client, "gpt-4o-mini", log_returns, top_examples, eliminated, prev_errors, feedback, NUM_FEATURES)

    if not feats:
        print("No valid features generated.")
        prev_errors += "No valid features.\n"
        continue

    results = []
    for f in feats:
        MAE, SP, NDCG, Score = evaluate_feature(f.flatten(), y_future)
        results.append((MAE, SP, NDCG, Score))
    df = pd.DataFrame(results, columns=["MAE", "Spearman", "nDCG", "Score"])
    print(df)

    # ---- Save round results ----
    save_path = os.path.join(SAVE_DIR, f"round_{rnd:02d}_features.json")
    to_save = []
    for i, name in enumerate(names):
        pattern = rf"def {name}\(.*?\):[\s\S]*?(?=def |\Z)"
        match = re.findall(pattern, code_str)
        code_part = match[0].strip() if match else f"# {name}: not found in raw_code"
        to_save.append({
            "name": name,
            "code": code_part,
            "MAE": float(df.iloc[i]["MAE"]),
            "Spearman": float(df.iloc[i]["Spearman"]),
            "nDCG": float(df.iloc[i]["nDCG"]),
            "Score": float(df.iloc[i]["Score"])
        })
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(to_save, f, indent=2, ensure_ascii=False)
    print(f"Saved generated features → {save_path}")

    # ---- Feedback Loop ----
    best_idx = df["Score"].idxmax()
    best_row = df.iloc[best_idx]
    mae_hist.append(best_row["MAE"])
    sp_hist.append(best_row["Spearman"])
    ndcg_hist.append(best_row["nDCG"])
    history.append((rnd, best_row["Score"]))
    feats_global.extend(feats)

    X_all = np.vstack(feats).T
    model = LGBMRegressor(n_estimators=80, learning_rate=0.05)
    model.fit(X_all, y_future[:len(X_all)])
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_all)
    importance = np.abs(shap_values).mean(0)
    importance /= importance.sum()
    importance_hist.append(importance.tolist())

    feedback_summary = "\n".join([
        f"{names[i]}: MAE={r[0]:.4f}, Spearman={r[1]:.4f}, nDCG={r[2]:.4f}, Score={r[3]:.4f}"
        for i, r in enumerate(results)
    ])
    feedback = f"{feedback_summary}\nFocus on improving features with higher Spearman and lower MAE."
    top_examples = code_str
    eliminated += ", ".join(names)
    prev_errors = "\n".join(errors)

print("\n===== Training Complete =====")


# ============================================================
# Visualization
# ============================================================
plt.style.use("seaborn-v0_8-whitegrid")

if history:
    rounds, scores = zip(*history)
    plt.figure(figsize=(6,4))
    plt.plot(rounds, scores, marker="o", linewidth=2)
    plt.title("AlphaQuant: Score Trend Across Rounds")
    plt.xlabel("Round"); plt.ylabel("Score")
    plt.tight_layout(); plt.show()

plt.figure(figsize=(6,4))
plt.plot(range(1,len(mae_hist)+1), mae_hist, color="tomato", marker="o")
plt.title("Figure 2: MAE Trend Across Iterations")
plt.xlabel("Iteration"); plt.ylabel("MAE")
plt.tight_layout(); plt.show()

if importance_hist:
    max_len = max(len(x) for x in importance_hist)
    arr = np.zeros((len(importance_hist), max_len))
    for i, imp in enumerate(importance_hist):
        arr[i, :len(imp)] = imp
    plt.figure(figsize=(7,4))
    plt.imshow(arr.T, aspect='auto', cmap='coolwarm', origin='lower')
    plt.colorbar(label="Normalized Importance")
    plt.title("Feature Importance Evolution (Smoothed Heatmap)")
    plt.xlabel("Round"); plt.ylabel("Feature Index")
    plt.tight_layout(); plt.show()

if len(sp_hist) > 3 and len(feats_global) > 3:
    feature_counts = np.arange(1, len(feats_global)+1)
    sharpe_vals = []
    for k in feature_counts:
        X_mat = np.vstack(feats_global[:k]).T
        y_target = y_future[:len(X_mat)]
        sharpe_vals.append(compute_test_sharpe(X_mat, y_target))
    sharpe_vals = np.array(sharpe_vals)
    corr_sp, _ = pearsonr(sp_hist[:len(sharpe_vals)], sharpe_vals[:len(sp_hist)])
    corr_ndcg, _ = pearsonr(ndcg_hist[:len(sharpe_vals)], sharpe_vals[:len(ndcg_hist)])

    fig, ax1 = plt.subplots(figsize=(6,4))
    ax2 = ax1.twinx()
    ax1.plot(feature_counts[:len(sp_hist)], sp_hist, color="royalblue", marker="o", label=f"Spearman (ρ={corr_sp:.2f})")
    ax1.plot(feature_counts[:len(ndcg_hist)], ndcg_hist, color="orange", marker="s", label=f"nDCG (ρ={corr_ndcg:.2f})")
    ax2.plot(feature_counts[:len(sharpe_vals)], sharpe_vals, color="green", linestyle="--", marker="x", label="Test Sharpe")
    ax1.set_xlabel("Number of Features")
    ax1.set_ylabel("Ranking Metrics")
    ax2.set_ylabel("Test Sharpe Ratio")
    plt.title("Figure 3: Spearman & nDCG vs Test Sharpe (More Features)")
    ax1.legend(loc="upper left"); ax2.legend(loc="upper right")
    plt.tight_layout(); plt.show()
