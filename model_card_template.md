# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model was developed for `educational purposes only` for the identified learner of Brannan Owens. 

**Last Updated**: 11/18/2024 **Model Version** : v1.0

**Model Type**: A Random Forest Classifier model was chosen for this project.

**Library**: The library list used to build this model includes: `scikit-learn`, `pandas`, `numpy`, and `fastapi` to name some of the primary requirements. The needed list can be found within the requirements.txt file that is included.

**Objective**: The goal is to predict whether an individual’s income exceeds $50K/year based on specific features using census data.  

**Input Features**: The training data included 6 numerical features and 8 categorical features that would assist in determining the target dependent variable.

**Target Variable**: The targeted dependent variable that the model was designed to predict is the salary variable which indicates (<=50K or >50K).

## Intended Use

**Primary Purpose**: The primary purpose of the model is to predict whether an individual’s income exceeds $50K/year based on demographic and employment data.

**Intended Users**: The intended Users can include Data Scientists, analysts, and policymakers exploring income distribution trends.

**Potential Use Cases**:
  - Identifying high-income individuals for targeted economic studies.
  - Analyzing income inequality across different demographics.
  - Assisting in resource allocation and public policy decisions.

<span style="color: red;">**Not Intended for**</span>: Making decisions that directly impact individuals' access to services or benefits without human oversight. To reiterate, this project is for educational purposes only and should not be used in any professional setting unless otherwise stated by the original source.

## Training Data

 **Source**: This model uses publicly available Census Bureau data forked from Git Hub https://github.com/udacity/Deploying-a-Scalable-ML-Pipeline-with-FastAPI

 **Description**: The dataset consists of demographic and employment information collected from the US Census Bureau.

 **Size**: The training set includes 26,048 rows of the original data set. (80% of the full dataset).

 **Features**:
  - Numerical: age, fnlgt, education-num, capital-gain, capital-loss, hours-per-week.
  - Categorical: workclass, education, marital-status, occupation, relationship, race, sex, native-country.

 **Target**: The target dependent variable is salary. (binary classification: <=50K or >50K).


## Evaluation Data

**Source**: This model uses publicly available Census Bureau data forked from Git Hub https://github.com/udacity/Deploying-a-Scalable-ML-Pipeline-with-FastAPI
**Description**: A subset of the same dataset used for training, held out for model evaluation.
**Size**: The test data includes 6,513 rows of the source data. (20% of the full dataset).
**Features**:
  - Numerical: age, fnlgt, education-num, capital-gain, capital-loss, hours-per-week.
  - Categorical: workclass, education, marital-status, occupation, relationship, race, sex, native-country.
**Target**: The target dependent variable is salary (binary classification: <=50K or >50K).

## Metrics

**Evaluation Metrics**: The metrics used to grade the model were Precision, Recall, and F1-score.

**Overall Model Performance**:
  - The overall model scored a precision of 0.7419.
  - The overall model scored a recall of 0.6384
  - The overall model scored a F1-Score of 0.6863.
- **Performance on Key Slices**:
  - Example: Workclass = "Private"
    - Precision: 0.7376 | Recall: 0.64.04 | F1-Score: 0.6856
  - Example: Education = "Bachelors"
    - Precision: 0.7523 | Recall: 0.7289 | F1: 0.7404


## Ethical Considerations

**Potential Biases**: 
  - The model may exhibit biases due to inherent biases in the dataset, particularly related to race, gender, and education.
  
**Risk of Misuse**: 
  - Predictions could be misinterpreted or misused in decision-making processes without proper human oversight.

**Mitigation Strategies**:
  - Evaluate model performance across different demographic slices to ensure fairness.
  - Include human review for decisions based on model outputs.
  - Regularly retrain the model with updated, balanced datasets to reduce bias over time.

## Caveats and Recommendations

**Caveats**:
  - The model’s predictions are based on historical census data, which may not fully represent current or future population trends.
  - Performance may degrade when applied to datasets with different distributions or unseen categories.

**Recommendations**:
  - Use the model as a supplementary tool, not as the sole basis for decision-making.
  - Regularly monitor model performance and update it with fresh data to maintain accuracy.
  - Avoid deploying the model in high-stakes environments without rigorous validation.
