# Improvement Plan — Customer Churn Prediction

> Generated: 2026-07-22
> Based on training run: XGBoost + class_weight, PR-AUC 0.6602

## Current State

| Metric | Value | Context |
|--------|-------|---------|
| PR-AUC | 0.6602 | Solid for churn on flat table (typical range: 0.55–0.75) |
| Best algorithm | XGBoost | Confirmed best for flat tabular data per research |
| Best strategy | class_weight | SMOTE underperformed (common with XGBoost + scaling pipelines) |
| Algorithms in config | LogisticRegression, RandomForest, XGBoost | LightGBM/CatBoost disabled in config; runtime container has `libgomp1` installed |
| Tuning | Optuna, 100 trials | 7-dimensional search space |

## Proposed Improvements (Ranked by Impact)

### Phase 1 — Quick Wins (~1-2h)

#### 1.1 Threshold Optimization
The 0.6602 PR-AUC is a curve-aggregated metric. At deployment, a specific threshold is chosen. Optimizing for F1-maximizing or cost-sensitive threshold yields better business metrics.

- Search thresholds 0.1–0.9 on the validation set
- Maximize F1 or a cost-weighted metric ($C_{\text{FN}} \approx 10 \times C_{\text{FP}}$)
- Current model likely using default 0.5 — suboptimal for 29% churn rate

#### 1.2 Deeper XGBoost Tuning & Early Stopping
Current space (7 params, 100 trials) samples ~14 points/dimension — sparse.

- Increase trials to 200–300
- Add `scale_pos_weight` as a tuneable parameter (currently computed as fixed neg/pos ratio)
- Set `eval_metric='aucpr'` during training and pass `eval_set` with `early_stopping_rounds=30` (speeds up Optuna by 3–5x and prevents overfitting)
- Add `reg_alpha` and `reg_lambda` to XGBoost search space (missing regularization)

### Phase 2 — Medium Effort (~4-8h)

#### 2.1 Feature Engineering Deep Dive & Pipeline Leakage Fix
Research consistently shows features > algorithms (15–25 AUROC point gap between weak and strong features).

**Pipeline Fixes:**
- **Quantile Leakage**: Replace live `.quantile()` calls in `add_business_flags` with static thresholds derived from `X_train` so inference/validation `transform()` calls don't leak or break single-row API calls.
- **Feature Scaling**: Update `Columns.CONTINUOUS` or `ColumnTransformer` to ensure newly engineered ratio features are scaled by `RobustScaler`.

**Interaction features:**
- `revenue × tenure` — high spend + short tenure = high risk
- `mou × eqpdays` — heavy usage on old equipment
- `custcare × months` — support calls relative to tenure
- `overage_ratio` = overage / mou (billing shock signal)
- `call_failure_rate` = dropblk / (peakvce + opeakvce) (network quality)
- `support_intensity` = (custcare + retcalls) / months (already exists as support_contact_rate, but ensure it's capturing spikes)

**RFM-style features:**
- `revenue_per_month` = revenue / months (already partial via revenue_per_mou, but add the tenure-normalized version)
- `avg_call_duration` = mou / (peakvce + opeakvce)
- `equipment_age` = eqpdays (already exists — could be binned differently)

**Business flags:**
- `is_billing_shock` = overage > revenue × 0.5
- `is_high_drops` = drop_call_rate > 0.1 (explicit threshold instead of quantile)
- `is_high_value_new` = is_high_value & tenure < 12mo (already exists as is_low_tenure_high_value)

#### 2.2 Enable LightGBM / CatBoost + Add Soft-Voting Ensemble
`libgomp1` is already installed in the Docker image (`Dockerfile`), so LightGBM/CatBoost can be re-enabled in `src/config.py` without environment crashes:

- Re-add LightGBM and CatBoost to `ALGORITHMS` in `src/config.py`
- Train XGBoost, LightGBM, and LogisticRegression independently
- Average their predicted probabilities (soft voting via `VotingClassifier`)
- Research shows ensembles add 1–3 AUROC points over single best model

### Phase 3 — Polish (Optional)

#### 3.1 Preprocessing Upgrades
- **Imputation**: Replace simple median with IterativeImputer or KNNImputer
- **Feature selection**: Mutual information or Boruta to reduce noise from 57+ features
- **Scaler**: Try QuantileTransformer (handles skewed distributions better than RobustScaler)
- **Encoding**: Target encoding for high-cardinality categoricals (none exist currently, but tenure_group/age_group could be ordinal-encoded instead of OHE)

#### 3.2 Advanced XGBoost
- **Cost-sensitive training**: Add instance-level weights (FP vs. FN cost ratios)
- **Focal loss**: Imbalance-XGBoost's focal loss for harder minority examples
- **Early stopping**: Use XGBoost's `early_stopping_rounds` with `eval_set` on validation

## Previous Run Results

### Algorithms × Strategies

| Algorithm | class_weight | SMOTE |
|-----------|-------------|-------|
| LogisticRegression | ✓ | ✓ |
| RandomForest | ✓ | ✓ |
| XGBoost | **0.6602** | ✗ (removed) |
| LightGBM | ✗ (errors) | ✗ (errors) |
| CatBoost | ✗ (errors) | ✗ (errors) |

### Best Result
- **Algorithm**: XGBoost
- **Strategy**: class_weight
- **PR-AUC**: 0.6602
- **Version**: 176 (Promoted to Production)

## Research Sources

- XGBoost confirmed as best flat-table algorithm for churn (Kumo.ai, Frontiers in AI 2026)
- Feature engineering yields 15–25 AUROC point improvements — the single biggest lever (Kumo.ai guide)
- SMOTE + XGBoost synergy well-documented; SMOTE + tuned XGBoost consistently outperforms class_weight alone (MDPI Technologies 2025)
- Soft-voting ensembles add 1–3 AUROC points (Frontiers in AI, Scientific Reports 2026)
- threshold optimization at ~0.53 improves business metrics significantly (Frontiers in AI 2026)
- `scale_pos_weight` tuning as hyperparameter recommended over fixed computation (XGBoost docs, arXiv 2025)
