# Remaining limitations

1. Population and currency: The model is trained on a public credit-risk dataset with USD-scale income and loan amounts. It is not validated for Indian (or other national) underwriting. Accepting INR-denominated numbers without a calibrated mapping is incorrect.

2. Fixed hyperparameters: C and l1_ratio are fixed, not grid-searched. A full nested CV search could improve or change the operating point.

3. Linear contributions are approximate: Feature drivers use centered linear contributions. They are not SHAP values and are not causal.

4. Threshold is F1-optimal on validation: Other business objectives (e.g. fixed recall, cost-sensitive thresholds) would choose a different cut-off. The threshold was not changed in this audit.

5. Missing rates: Interest rate and employment length missingness is handled by median imputation learned on training data only. Systematic differences in missingness at serving time can still cause skew.

6. No fairness audit: Group fairness across protected attributes is out of scope for this package.

7. Decision support only: The UI explicitly recommends manual review; it is not an automated approve/reject system.

8. sklearn version: The artifact was trained with scikit-learn 1.9.x. Loading under substantially older versions may emit warnings or fail. Requirements pin a compatible range.
