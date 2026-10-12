# Changelog

## Fixes and improvements (final project package)

### Bugs fixed
- **Feature-means key mismatch**: train.py saved feature_means; app.py loaded means, so contribution centering always fell back to uncentered values. App now loads feature_means (with legacy means fallback). The serialized artifact was also patched to include both keys without retraining.
- **Undefined MODEL_LOAD_ERROR**: Always initialized so failure paths are safe.
- **Out-of-distribution inputs**: Validation limits for income and loan amount were far wider than the training support (max loan in data = 35,000). Extreme values (e.g. loan 200,000) produced near-zero probabilities via extreme standardized features. Limits are now aligned with the data distribution with modest headroom, and error messages explain why.
- **Loan exceeding income**: Added an explicit check that loan_percent_income is at most 1.0.
- **README pipeline description**: Previously claimed 5-fold CV tuning; the implementation uses fixed C=1.0, l1_ratio=0.5 and a dedicated validation split only for threshold selection. Documentation now matches the code.

### UI / UX
- Clearer field hints with training-scale ranges.
- Accessibility: labels linked to inputs, focus styles, aria-live for errors and results, button types.
- Currency/scale disclaimer: does not claim suitability for Indian lending or any specific currency.
- Presets remain realistic and inside training support; the model alone determines outcomes.
- Drivers section shows a fallback message if contributions are unavailable.

### Engineering
- Pinned dependency ranges compatible with the saved sklearn 1.9.1 artifact.
- Automated pytest coverage for validation, metrics API, prediction API, presets, and OOD rejection.
- Dual key (feature_means / means) written by train.py for future retrains.

### Unchanged (by design)
- Model architecture, hyperparameters, threshold (0.37), preprocessing pipeline, and reported test metrics.
- Dataset and evaluation methodology.
