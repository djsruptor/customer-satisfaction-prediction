# Santander Customer Satisfaction Prediction
This project was built using the dataset from the [Santander Customer Satisfaction 2016 Kaggle competition](https://kaggle.com/competitions/santander-customer-satisfaction) to identify customers who are more likely to be dissatisfied with the bank's service before they raise a complaint or leave.


## 1. Business Problem 

### Problem Description
>From frontline support teams to C-suites, customer satisfaction is a key measure of success. Unhappy customers don't stick around. What's more, unhappy customers rarely voice their dissatisfaction before leaving.
>
>Santander Bank is asking for help to identify dissatisfied customers early in their relationship. Doing so would allow Santander to take proactive steps to improve a customer's happiness before it's too late.

- *Santander Customer Satisfaction, Kaggle*

### Dataset

The dataset contains over 76.000 clients and 369 anonymized numerical features.

Each row represents a single customer. The target variable (`TARGET`) is binary:
- `0` = Satisfied
- `1` = Dissatisfied

Source: [OpenML dataset 46859](https://www.openml.org/d/46859) / Kaggle Santander Customer Satisfaction competition.

## 2. Data Preparation and EDA
The `download_data.py` script retrieves the dataset using `OpenML` API and saves it as a raw Parquet file. 

**EDA and pre-processing steps**(see `notebook.ipynb`)**:**
- Removed highly correlated features, retaining one feature from each correlated pair arbitrarily because the variables are anonymized.
- The target is highly imbalanced: 96% satisfied and 4% dissatisfied.
- Data was split into train/validation/test (60/20/20) using `sklearn.model_selection.train_test_split`

<img width="555" height="443" alt="Screenshot 2026-10-06 at 10 50 36 PM" src="https://github.com/user-attachments/assets/3dfde1fb-2c21-4471-b648-3b1ecfa566cf" />

## 3. Modelling 
**Selection, Training, and Tuning**

Model selection and tuning are documented in `notebook.ipynb`, with final training logic in `train.py`. 

Models evaluated:
- `LogisticRegressionCV`
- `DecisionTreeClassifier`
- `RandomForestClassifier`
- `XGBClassifier`

Performed hyperparameter tuning using `GridSearchCV` and `RandomizedSearchCV`, final results were stored and compared in a consolidated performance table.

|model                 |val_roc  |test_roc |val_pr   |test_pr  |roc_var  |pr_var    |rank   |
|----------------------|---------|---------|---------|---------|---------|----------|-------|
|**XGBClassifier**     |**0.834**|**0.845**|**0.195**|**0.193**|**0.011**|**-0.002**|**1.0**|
|RandomForestClassifier|0.521    |0.830    |0.043    |0.192    |0.309    |0.149     |2.0    |
|DecisionTreeClassifier|0.812    |0.826    |0.155    |0.156    |0.014    |0.001     |3.0    |
|LogisticRegressionCV  |0.793    |0.803    |0.136    |0.146    |0.010    |0.010     |4.0    |

## 4. Performance Validation

Additional validation confirmed model stability:
- Re-trained on full_train(train + validation) and evaluated on remaining 20%
- Computed confusion matrix and classification report
- Plotted ROC, PR, Precision, Recall and F1 curves
- Verified no single feature contributed over 40% of total importance

<img width="758" height="251" alt="Screenshot 2026-10-06 at 11 01 36 PM" src="https://github.com/user-attachments/assets/c4560ab1-f113-494e-ae4d-338ca4ac5c40" />

Using a default 0.5 threshold would miss too many dissatisfied customers. Therefore, I set the threshold at 0.136 by maximizing F1 on the validation set, which also increased recall of dissatisfied customers.


## 5. Deployment
A trained model is served through FastAPI via (`predict.py`).

There are two endpoints:
- `GET /` - Returns a JSON sample of feature values for a given index
- `POST /predict` - Returns satisfaction prediction based on a customer number inserted by the user input

**Reproducibility**

To reproduce:
```bash
# 1. Environment setup
pip install uv

uv venv .venv
source .venv/bin/activate

uv sync

# 2. Download data
uv run python scripts/download_data.py

# 3. Train model
uv run python train.py

# 4. Run API
uv run uvicorn predict:app --reload

# 5. Deploy and run Docker container
docker build -t santander-service .
docker run -p 9696:9696 santander-service
``` 

## 6. What I learned
The main lesson from this project was that EDA should shape the modeling strategy, not just describe the dataset. In this case, the 4% positive class made class imbalance the defining constraint of the problem, so evaluation design and threshold selection mattered as much as model choice.

If I extended the project, I would spend more time on:
- calibration
- cost-based thresholding
- SHAP or other explainability methods
- drift monitoring
- a clearer operational definition of what action follows a positive prediction

## Citation
@misc{santander-customer-satisfaction,
    author = {Soraya_Jimenez and Will Cukierski},
    title = {Santander Customer Satisfaction},
    year = {2016},
    howpublished = {\url{https://kaggle.com/competitions/santander-customer-satisfaction}},
    note = {Kaggle}
}

## License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
