# src/api/schemas.py
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict

YesNo = Literal["Yes", "No"]
YesNoPhone = Literal["Yes", "No", "No phone service"]
YesNoNet  = Literal["Yes", "No", "No internet service"]

class TelcoInput(BaseModel):
    model_config = ConfigDict(extra="ignore")  # 有多余字段时忽略，不报错

    gender: Literal["Male", "Female"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=72)

    PhoneService: YesNo
    MultipleLines: YesNoPhone

    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: YesNoNet
    OnlineBackup: YesNoNet
    DeviceProtection: YesNoNet
    TechSupport: YesNoNet
    StreamingTV: YesNoNet
    StreamingMovies: YesNoNet

    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]

    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)
