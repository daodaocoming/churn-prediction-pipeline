cat > reports/model_compare.md <<'MD'
# Model Comparison — Day 18 ➜ 20

| Model | ROC-AUC | PR-AUC |
|-------|:------:|:------:|
| Logistic Regression | 0.852 | 0.666 |
| **XGBoost (Optuna)** | **0.872** | **0.712** |
| LightGBM (Optuna) | 0.857 | 0.683 |

<div align="center">

![ROC curves](notebooks/figures/roc_compare.png)  
*Figure 1 — ROC curve comparison*

![PR curves](notebooks/figures/pr_compare.png)  
*Figure 2 — Precision-Recall curve comparison*

</div>

---

## Final decision

* **Champion:** **XGBoost (Optuna tuned)**  
* **Saved as:** `models/final/model.pkl`  
* **Why:** highest PR-AUC (+0.029 vs LGBM, +0.046 vs LogReg) with sub-5 ms inference.

Use it anywhere:

```python
from src.models.final_model import load
model = load()

