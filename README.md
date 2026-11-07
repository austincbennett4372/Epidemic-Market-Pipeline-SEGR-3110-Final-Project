# Epidemic Market Entropy Pipeline

**Authors:** Austin Bennett & Mahad Hayat
**Course:** SEGR 3110 (Systems Design and Deployment), UNC Charlotte | **Date:** May 2026
**Language:** R (~5,400 lines), with Python/TensorFlow through `reticulate`

## Summary

This project tests whether the statistical "signature" of epidemic growth (COVID-19, dengue, TB, influenza, and nine other diseases) resembles, or can help predict, the behavior of US equity index returns (S&P 500 and Dow Jones). We built one end-to-end pipeline that pulls every series live, extracts rolling-window features, and then examines the relationship through four independent methods: SVM classification, Gaussian mixture regime detection, Markov eigenanalysis, and LSTM forecasting with a Bayesian update.

**Result:** no consistent pattern or correlation was found between fluctuations in disease case counts and movements in the S&P 500 or DJIA. Disease features did not beat a naive baseline at predicting market direction, and adding them made the LSTM forecasts worse. The pipeline did surface one robust result about the market itself: the Sell regime is never sustained for consecutive periods. The full result is discussed under [Findings](#findings).

## Key Contributions

- **Built an end-to-end analysis pipeline covering 13 disease series and 2 US indices** by integrating feature engineering, SVM classification, GMM regime detection, Markov reduction, and LSTM forecasting into a single script that runs top to bottom from raw downloads to final figures.
- **Automated ingestion of live data from 6 public providers** (Yahoo Finance, Our World in Data, OpenDengue, WHO GHO, PAHO, CDC) by writing per-disease loaders with ordered fallback sources, so that a single unreachable source skips that series instead of breaking the run.
- **Aligned daily market data with weekly and irregular surveillance data** by weekly-aligning every series to a Saturday week end, interpolating gaps, and running an alignment diagnostic that measures index × disease overlap before modeling.
- **Showed that COVID-19 growth and index returns have distinct statistical fingerprints**, measured by 100% (SPX) and 96.9% (DJIA) held-out SVM accuracy with permutation p ≤ 0.005 on both, by tuning polynomial-kernel SVMs with 5-fold cross validation on a time-ordered 75/25 split and running a 199-permutation test in parallel across all CPU cores.
- **Tested whether disease features could predict next-week market direction across every disease and both indices**, measured against an always-predict-up baseline, by training RBF-kernel SVMs on the combined disease–market feature space and projecting the decision boundaries into PCA space.
- **Identified 13 distinct market regimes without assuming a fixed number of states** by fitting Gaussian mixture models with 2–13 components (VVV covariance selected by BIC), then validating the fit with silhouette scores, a K-means elbow plot, and a degeneracy check on component weights and posteriors.
- **Reduced the 13×13 regime dynamics to an interpretable 4-state Buy/Hold/Watch/Sell Markov chain**, which showed the Sell regime has a self-transition probability of 0.0000, by solving for the stationary distribution (π·P = π) and aggregating states weighted by stationary mass, with 3- and 5-state reductions kept for comparison.
- **Benchmarked a 3-layer LSTM regime classifier on 4 datasets at 4 forecast horizons** (k = 1, 4, 8, 12) against a persistence baseline and the 25% chance level. Before training on real data, the architecture was calibrated on a Mackey-Glass chaotic series matched to the SPX entropy embedding signature (average mutual information and Cao's method), which confirmed it could recover structure where structure was known to exist.
- **Combined the Markov transition prior with the LSTM softmax output** in a Bayesian update to produce regime posteriors and a P(Sell) time series, so that the forecasting and Markov stages could be compared on the same footing.

## Findings

The central finding of this project is that no consistent pattern or correlation was found between the fluctuation in case counts of the diseases studied and the movement of the S&P 500 or DJIA. This held across diseases, across both indices, and across the different methods used to look for a relationship.

Note that this is a null result; the pipeline was designed so that a real relationship would have had several chances to show itself, and the evidence from each stage points the same direction.

### 1. Separability: the two domains look different, not alike

| Pairing (daily, polynomial kernel) | Test accuracy | Permutation p-value (199 permutations) |
|---|---|---|
| COVID-19 vs SPX | 1.000 | ≤ 0.005 |
| COVID-19 vs DJIA | 0.969 | ≤ 0.005 |

The SVM separates COVID-19 features from index features almost perfectly, and the permutation test confirms this is not chance. What stands out is what this result means: high separability indicates that disease growth and market returns have *different* statistical fingerprints. If epidemic dynamics shared a signature with market behavior, the two would be harder to tell apart, not easier.

Note that with 199 permutations the smallest p-value that can be resolved is 1/200 = 0.005, so both results are reported as p ≤ 0.005.

### 2. Prediction: disease features did not beat a naive baseline

| Pairing (weekly, RBF kernel) | n (test) | Accuracy | Always-predict-up baseline |
|---|---|---|---|
| TB → SPX direction | 267 | 0.588 | 0.569 |
| Dengue → DJIA direction | 426 | 0.547 | 0.554 |
| Cholera → DJIA direction | 58 | 0.569 | 0.569 |

When the question changed from "are these different?" to "can disease features predict next week's market direction?", none of the pairings meaningfully beat a model that simply predicts "up" every week. Dengue vs DJIA scored *below* that baseline, and the Cholera model predicted the same class for every observation, which means it learned nothing beyond the class proportions. TB vs SPX was about 2 percentage points above the baseline on 267 observations, which is within what would be expected from noise.

### 3. Forecasting: adding disease data made the LSTM worse

| LSTM dataset | Skill (k = 4 vs persistence) | Accuracy (k = 1 vs 25% chance) |
|---|---|---|
| Mackey-Glass (control) | PASS | PASS |
| SPX only | PASS | PASS |
| DJIA only | PASS | FAIL |
| Combined epidemic–market | **FAIL** | **FAIL** |

The Mackey-Glass control passed, which confirms that the architecture can recover structure from a chaotic signal when that structure exists. The market-only models also passed at least one criterion. The combined epidemic–market model was the only one to fail both, which indicates that the disease features added noise rather than information.

### 4. What *was* found: market regime structure

The one robust structural result is about the market itself, independent of the disease data. In the 4-state Markov reduction, the Sell regime has a self-transition probability of 0.0000:

| From \ To | Buy | Hold | Watch | Sell |
|---|---|---|---|---|
| Buy | 0.3285 | 0.2682 | 0.3614 | 0.0419 |
| Hold | 0.2877 | 0.4934 | 0.1998 | 0.0191 |
| Watch | 0.4314 | 0.2184 | 0.3427 | 0.0075 |
| Sell | 0.6346 | 0.2596 | 0.1058 | **0.0000** |

The market enters Sell but never stays there for consecutive periods, and it exits to Buy about 63% of the time. We interpret this as Sell acting as a short-lived perturbation rather than a stable state.

### Conclusion

Across separability, direction prediction, and sequence forecasting, the disease data never produced a signal that held up against a simple baseline. This leads to the conclusion that, at the daily and weekly resolutions tested and at the national case-count level, epidemic growth dynamics do not provide a usable signature for characterizing or predicting US index returns.

## Pipeline

### Stage 0: Environment setup
- Installs TensorFlow 2.18.0 into the `r-tensorflow` environment through `reticulate` if it is missing.
- Installs and loads any missing CRAN packages (`keras3`, `tidyverse`, `e1071`, `mclust`, `quantmod`, `forecast`, `plotly`, and others).
- Sets `set.seed(1296)` and disables scientific notation. `PRINT_INTERMEDIATE_PLOTS` toggles intermediate diagnostic plots.

### Stage 1: Data collection, feature extraction, SVM feasibility
| Section | Content |
|---|---|
| 1 | Starts a parallel PSOCK cluster on all detected cores and seeds each worker's RNG. |
| 2 | Feature functions. `window_features()` computes **mean, sd, lag-1 autocorrelation, skewness, kurtosis, Shannon entropy, entropy², and sd × entropy** over rolling windows. Entropy is calculated in nats from a 20-bin histogram of the window, and at least 10 valid observations are required. `clean_covid_style_growth()` aligns a case series to weeks ending Saturday, interpolates gaps, and computes log growth. |
| 3a/3b | Downloads S&P 500 (`^GSPC`) and DJIA (`^DJI`) from Yahoo Finance and builds daily and weekly log returns. |
| 4a–4d | COVID-19 (OWID, US) vs SPX and DJIA at daily frequency. Polynomial-kernel SVM tuned with `tune.svm` (5-fold CV) on a 75/25 time split, followed by a 199-permutation test. |
| 5a/5b | Dengue (OpenDengue national extract) vs SPX and DJIA at weekly frequency. |
| 6–7 | Loaders for the remaining diseases, combined into one disease panel. |
| 8 | Alignment diagnostic: confirms all dates fall on Saturday and measures index × disease overlap. |
| 9–15 | SVM pipeline across all diseases for SPX (9–12) and DJIA (13–15), with feature visualizations, cross-disease comparisons, SVC decision boundaries, and SVR epsilon-tube plots. |
| 16 | SPX vs DJIA cross-index comparison. |
| 17 | Exports all CSVs. |

### Stage 2: GMM regime detection
- Combines the SPX × TB and DJIA × Dengue feature sets and fits `mclust::Mclust` with G = 2…13, selected by BIC. The selected model is VVV (each component has its own volume, shape, and orientation) with 13 components.
- Validates with sampled silhouette, a BIC plot, a K-means elbow plot, and a degeneracy check.
- Collapses clusters into **Buy / Hold / Sell** by mean-return tercile.
- Produces feature distributions by regime, 2D scatter with 1σ ellipses, and 3D mixture density plots.
- Computes regime entropy and the **regime transition matrix** used in later stages.

### Stage 3: Eigenanalysis and Markov reduction
- Row-normalizes and validates the GMM transition matrix, then solves for the **stationary distribution** (π·P = π).
- Reduces it to a **4×4** (Buy/Hold/Watch/Sell) matrix, plus **3×3** and **5×5** matrices for comparison. Each reduction is weighted by stationary mass over the grouped states.

### Stage 4: LSTM forecasting and Bayesian updating
- **Regimes:** entropy thresholds of 0.32 / 0.60 / 0.87 define four classes: **Converging, Transitional, Diverging, Perturbation**. Defaults are `K = 28` and `LOOKBACK = 520`.
- **Model** (`build_mdn_model`):

  | Layer | Units | Activation | Regularization |
  |---|---|---|---|
  | LSTM 1 | 256 | tanh | L2 = 0.01, dropout = 0.15 |
  | LSTM 2 | 128 | tanh | L2 = 0.01, dropout = 0.15 |
  | LSTM 3 | 64 | tanh | L2 = 0.01, dropout = 0.15 |
  | Dense | 16 | Leaky ReLU | |
  | Output | 4 | softmax | |

  Trained with Nadam (lr = 0.0003) and categorical cross-entropy, with early stopping and learning-rate reduction on plateau. The function name is a holdover from an earlier mixture-density-network design.
- **Class imbalance:** optional minority-class oversampling and inverse-frequency sample weights.
- **Controls:** a Mackey-Glass chaotic series (`mackey_glass`) and a Takens-style `embedding_calibration` to choose lag and embedding dimension.
- **`run_full_pipeline()`** trains and evaluates on four datasets (Mackey-Glass, combined dengue/SPX/DJIA, SPX only, DJIA only) and reports MAE vs persistence (skill score), accuracy, macro-F1, and per-class F1 at k = 1, 4, 8, 12.
- **Validity verdict:** PASS requires k = 4 skill > 0 and k = 1 accuracy > 0.25 (4-class chance).
- **Bayesian layer:** combines the Markov prior with the LSTM likelihood into regime posteriors for k = 4, 8, 12, and produces a posterior heatmap and a P(Sell) time series.

## Data Sources

All series are downloaded at runtime.

| Series | Source |
|---|---|
| S&P 500, DJIA | Yahoo Finance via `quantmod` |
| COVID-19 (US) | Our World in Data (GitHub raw CSV) |
| Dengue | OpenDengue national extract v1.3 |
| TB, malaria, measles, cholera, Ebola, Chagas | WHO GHO API, OWID, PAHO (multiple fallbacks per disease) |
| Influenza (US) | `cdcfluview`, falling back to the CDC data API |
| Mpox | OWID / WHO |
| Pertussis, hepatitis A, leishmaniasis | CDC and other loaders (Stage 1, Section 6) |

## Requirements
- R with permission to install packages (the script installs what it needs).
- Python and TensorFlow 2.18.0, installed automatically through `reticulate` into `r-tensorflow`.
- An internet connection.
- Enough RAM and time for LSTM training. Parallel stages use all CPU cores by default (`USE_ALL_CORES <- TRUE`).

## How to Run
1. **Set the working directory.** Line 63 (`setwd(...)`) points to the original project folder; change it to the folder where you want outputs written.
2. Start a fresh R session. Stage 0 must run before `keras3` is loaded.
3. Run the script **top to bottom, in order**. Later stages read files written by earlier ones (for example, Stage 2 reads `model_features_spx_TB_GLOBAL_weekly_FINAL.csv`), and Stage 4 reuses objects created in memory by earlier stages.

## Outputs

| Stage | Files |
|---|---|
| 1 | `model_features_*_FINAL.csv`, `paired_*_CLEAN_FINAL.csv`, `svm_accuracy_by_disease_{SPX,DJIA}_FINAL.csv`, `svm_cross_index_comparison_FINAL.csv` |
| 2 | `gmm_transition_matrix_csv_FINAL.csv`, `gmm_transition_matrix_FINAL.rds`, `gmm_clustered_full_FINAL.rds`, `gmm_soft_assignments_FINAL.rds`, `gmm_diagnostics.png`, `gmm_final_2d.png` |
| 3 | `market_4_regime_transition_matrix_FINAL.csv`, `proof_for_comparison_market_{3,5}_regime_transition_matrix_FINAL.csv` |
| 4 | `embedding_calibration_FINAL.csv`, `results_real_v14_LEAN_2Layer.rds`, `lstm_full_evaluation_summary_v14.csv` |

## Notes and Limitations
- **Live data:** results reflect whatever data is available on the day the script is run, so exact numbers can shift between runs.
- **Reproducibility:** the seed is fixed at `1296`; however, TensorFlow training and parallel workers can still introduce small run-to-run differences.
- **Resolution:** the analysis was run at daily and weekly frequency on national-level case counts. A relationship at a finer resolution (for example, sector-level returns or regional outbreaks) was not tested and is outside the scope of this project.
- **Flu data:** to use the `cdcfluview` package instead of the API fallback, run `install.packages("cdcfluview")`.
