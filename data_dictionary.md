| Field Name       | Data Type   | Example Value   | Description / Business Meaning
|:-----------------|:------------|:----------------|:------------------------------------------------------
| customerID       | string      | 7590-VHVEG      | Customer ID
| gender           | category    | Female          | Whether the customer is a male or a female
| SeniorCitizen    | int (0/1)   | 0               | Whether the customer is a senior citizen or not (1, 0)
| Partner          | category    | Yes             | Whether the customer has a partner or not (Yes, No)
| Dependents       | category    | No              | Whether the customer has dependents or not (Yes, No)
| tenure           | int         | 1               | Number of months the customer has stayed with the company
| PhoneService     | category    | No              | Whether the customer has a phone service or not (Yes, No)
| MultipleLines    | category    | No phone…       | Whether the customer has multiple lines or not (Yes, No, No phone service)
| InternetService  | category    | DSL             | Customer’s internet service provider (DSL, Fiber optic, No)
| OnlineSecurity   | category    | No              | Whether the customer has online security or not (Yes, No, No internet service)
| OnlineBackup     | category    | Yes             | Whether the customer has online backup or not (Yes, No, No internet service)
| DeviceProtection | category    | No              | Whether the customer has device protection or not (Yes, No, No internet service)
| TechSupport      | category    | No              | Whether the customer has tech support or not (Yes, No, No internet service)
| StreamingTV      | category    | No              | Whether the customer has streaming TV or not (Yes, No, No internet service)
| StreamingMovies  | category    | No              | Whether the customer has streaming movies or not (Yes, No, No internet service)
| Contract         | category    | Month-to-…      | The contract term of the customer (Month-to-month, One year, Two year)
| PaperlessBilling | category    | Yes             | Whether the customer has paperless billing or not (Yes, No)
| PaymentMethod    | category    | Electronic…     | The customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
| MonthlyCharges   | float       | 29.85           | The amount charged to the customer monthly
| TotalCharges     | float       | 29.85           | The total amount charged to the customer
| Churn            | category    | No              | Whether the customer churned or not (Yes or No)

---

**Version:** 1.0 | Author: *Doris Cai* | Created: *2025‑07‑16*