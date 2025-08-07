# Model Card · XGBoost Telco-Churn Classifier

| Item | Detail |
|------|--------|
| **Version** | 1.0 · 2025-08-06 |
| **Owners** | Doris |
| **Algorithm** | XGBoost (Optuna-tuned, binary:logistic) |
| **Training Data** | 7 043 rows · IBM Telco Customer-Churn dataset (cleaned, winsorized, engineered features × 38) |
| **Metrics (5-fold CV)** | ROC-AUC **0.872** · PR-AUC **0.712** |
| **Intended Use** | Rank current subscribers by churn risk →  prioritise retention/offers |
| **Limitations** | Dataset is US-telco specific; not calibrated for other industries or geographies |
| **Ethical Considerations** | • Do **not** withhold service based solely on prediction<br>• Monitor false-positive impact on senior & low-income segments |
| **Training Script** | `src/train/tune_xgb_optuna.py` |
| **Source Repo** | `github.com/<your-user>/churn-prediction-pipeline` |

---

## 1 · Global Explainability

| Plot | Description |
|------|-------------|
| ![SHAP summary](../figures/shap_summary.png) | SHAP dot summary（38 features） |
| ![Top-20 bar](../figures/feature_importance_bar.png) | AVERAGE \|SHAP\| Top-20 FEATURES |

*Key drivers*  
1. `Is_MonthToMonth_1.0` (+0.54) – month-to-month contracts greatly increase risk  
2. `tenure` (-0.43) – longer tenure lowers probability  
3. `InternetService_Fiber optic` (+0.31) – fibre users churn more, likely price-sensitive  

---

## 2 · Local Explanations (3 typical customers)

| ID | Waterfall / Force plot | Pred-prob |
|----|------------------------|-----------|
| 1 · Low risk | ![Client 1](../figures/client_1_force.png) | **7 %** |
| 2 · Borderline | ![Client 2](../figures/client_2_force.png) | **53 %** |
| 3 · High risk | ![Client 3](../figures/client_3_force.png) | **91 %** |

---

## 3 · Deployment Notes

- **Champion artefact** `models/final/model.pkl` (joblib)  
- **Feature pipeline** `models/feature_pipeline_v2.pkl`  
- **REST endpoint** `POST /predict` → returns probability + top-3 SHAP features  


