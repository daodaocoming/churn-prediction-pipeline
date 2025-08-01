# Logistic Regression Baseline (Day 14)

Dataset | Telco Clean (7,043 rows × 38 features)  
Cross-validation | 5-fold StratifiedKFold (random_state = 42)  
Parameters | `class_weight='balanced', max_iter=1000`  
Model file | `models/logreg.pkl`

| Metric | Mean | Std |
|--------|-----:|----:|
| **ROC-AUC** | **0.848** | 0.011 |
| **PR-AUC**  | **0.661** | 0.015 |

---

## Optimal Threshold (max F1)

Threshold search based on `precision_recall_curve` + F1 formula:

| Threshold | F1 | Precision | Recall |
|-----------|----|-----------|--------|
| **0.59**  | **0.643** | 0.579 | 0.722 |

> *Explanation*  
> - **Precision 0.579**: At this threshold, 57.9% of predicted Churn cases are actual churners.  
> - **Recall 0.722**: Captures 72.2% of actual churners.  
> - **F1 0.643**: Achieves a balance between Precision and Recall.

### Confusion-Matrix-Style Summary (on training set)

|                  | Pred No-Churn    | Pred Churn         |
|------------------|------------------|--------------------|
| **Actual No-Churn** | Precision 0.89, Recall 0.81 | |
| **Actual Churn**    | | *F1 0.64* (Precision 0.58 / Recall 0.72) |

---

### Recommendation

For *production use*, if **recall is prioritized** (to avoid missing potential churners), use the **0.59** threshold.  
If business focus shifts toward **precision**, adjust the threshold along the PR curve in the Notebook and recompute metrics accordingly.
