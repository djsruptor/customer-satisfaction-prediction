# Santander Customer Satisfaction Prediction
This project was built using [Santander Customer Satisfaction 2016 Kaggle competition](https://kaggle.com/competitions/santander-customer-satisfaction) with the objective of identifying customers who are more likely to be upset with the bank's service. 


## 1. Business Problem Description
From frontline support teams to C-suites, customer satisfaction is a key measure of success. Unhappy customers don't stick around. What's more, unhappy customers rarely voice their dissatisfaction before leaving.

Santander Bank is asking for help to identify dissatisfied customers early in their relationship. Doing so would allow Santander to take proactive steps to improve a customer's happiness before it's too late.

A dataset with over 76k clients and 360 anonymized features is provided.

Each record represents a single customer and 369 numeric features. The target variable (`TARGET`) is binary:
`0` = Satisfied
`1` = Dissatisfied


## 2. Data Preparation and EDA

The `download_data.py` script retrieves the dataset using `openml` API and save it as a parquet file. 

EDA and pre-processing steps (see `notebook.ipynb`):
- Confirmed no missing values -> No imputation required
- All columns are numerical -> no categorical encoding required
- Identified highly correlated pairs and reduced features (369 -> 203) to improve efficiency
- Target imbalanced (`0`: 96%, `1`: 4%) visualized with bar chart
- Data was split into train/val/test (60/20/20) using `sklearn.model_selection.train_test_split`

## Model Selection, Training, and Tuning

Model selection and tuning are documented in `notebook.ipynb`, with final training logic in `train.py`. 

Models evaluated:
- `LogisticRegressionCV`
- `DecisionTreeClassifier`
- `RandomForestClassifier`
- `XGBClassifier`

5-fold cross-validation was implemented using `GridSearchCV` and `RandomizedSearchCV` for parameter optimization.
Results were stored and compared using a consolidated performance table.

> While ROC-AUC provides a global ranking metric, PR-AUC better reflects the model's ability to identify dissatisfied customers — a priority for marketing and customer success teams who act on these predictions. For this reason, both metrics were taken into account when evaluating models performance.

**Best performing model was XGBoost:**
- ROC-AUC: 0.848
- PR-AUC: 0.194

## Performance Validation

Additional validation confirmed model stability and fairness:
- Re-trained on full_train(train+validation) and evaluated on remaining 20%
- Computed confusion matrix and classification report
- Determined optimal F1 threshold for better TPR: **0.136**
- Plotted ROC, PR, Precision, Recall and F1 curves
- Verified no single feature contributed over 40% of total importance

> In this context, optimizing the F1-based threshold and prioritizing a higher TPR is essential. Retaining dissatisfied customers has a disproportionately positive business impact, so accepting a moderate increase in false positives (retaining already-satisfied customers) is a reasonable trade-off for capturing more truly dissatisfied ones.

## Model Deployment

A trained model is served through FastAPI via (`predict.py`).

There are two endpoints:

- `GET/` - Returns a JSON sample of feature values for a given index
- `POST/predict` - Returns satisfaction prediction based on customer number (user input) and threshold

## Reproducibility

- Dataset source: [OpenML 46859](https://www.openml.org/d/46859)
- Size: ~76k rows x 369 features

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
