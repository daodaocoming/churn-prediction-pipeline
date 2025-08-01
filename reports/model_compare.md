# Model Comparison (Day 18)

The plots below compare the 5-fold CV performance of the two candidates on the **clean Telco dataset (7 043 rows × 38 features)**:

<div align="center">

| ![ROC Curve](figures/roc_compare.png) | ![PR Curve](figures/pr_compare.png) |
|:--:|:--:|
| **ROC‐AUC** | **PR‐AUC (Average Precision)** |

</div>

<br>

| Model   | ROC-AUC | PR-AUC |
|---------|:------:|:------:|
| **LogReg** | 0.852 | 0.666 |
| **XGB**    | **0.872** | **0.712** |

---

### Key take-aways

* **XGBoost outperforms Logistic Regression** by&nbsp;`+0.020` ROC-AUC and `+0.046` PR-AUC.  
  The lift is consistent across the full curve, not just at a single threshold.
* On the PR plot, the orange XGB curve stays above the blue LogReg curve for most recall levels—  
  indicating higher precision, especially in the high-recall region important for churn mitigation.
* **Decision:** Promote XGBoost as the default production model.  
  Artifact path → `models/xgb_optuna_best.pkl`
