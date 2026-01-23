# Banking_Customer_Churn

## Step 1 : Dataset choice
a. Bank Customer Churn Dataset
- The dataset contains information about bank customers and whether they have Exited (i.e., closed their accounts) or not.
     Feature Size: 12
     Records: 10000

## Step 2 : Evaluation Metric

  ### "Model": "LogisticRegression",
  - "Accuracy": 0.706,
  - "Precision": 0.3820078226857888,
  - "Recall": 0.7199017199017199,
  - "F1": 0.4991482112436116,
  - "MCC": 0.34970214776872216,
  - "AUC": 0.7737706890249264,
  ![alt text](outputs/plots/logisticregression.png)

  ### "Model": "RandomForestClassifier",
  - "Accuracy": 0.821,
  - "Precision": 0.5513626834381551,
  - "Recall": 0.6461916461916462,
  - "F1": 0.5950226244343891,
  - "MCC": 0.4835506638092883,
  - "AUC": 0.8486174926852892,
  ![alt text](outputs/plots/randomforest.png)

  ### "Model": "DecisionTreeClassifier",
  - "Accuracy": 0.8275,
  - "Precision": 0.5665236051502146,
  - "Recall": 0.6486486486486487,
  - "F1": 0.6048109965635738,
  - "MCC": 0.49698129922639683,
  - "AUC": 0.8329446549785534,
    ![alt text](outputs/plots/decisiontree.png)

    ### "Model": "GaussianNB",
    - "Accuracy": 0.7835,
    - "Precision": 0.3142857142857143,
    - "Recall": 0.05405405405405406,
    - "F1": 0.09224318658280922,
    - "MCC": 0.052405760867435834,
    - "AUC": 0.7439334557978626,
    ![alt text](outputs/plots/gaussiannb.png)

    ### "Model": "KNN   ",
    - "Accuracy": 0.8265,
    - "Precision": 0.7027027027027027,
    - "Recall": 0.25552825552825553,
    - "F1": 0.3747747747747748,
    - "MCC": 0.3505195113984483,
    - "AUC": 0.7757989113921318,
    ![alt text](outputs/plots/knn.png)

    ### "Model": "XGBoost",
    - "Accuracy": 0.853,
    - "Precision": 0.7383966244725738,
    - "Recall": 0.42997542997543,
    - "F1": 0.5434782608695652,
    - "MCC": 0.4871276453473892,
    - "AUC": 0.8401267214826538,
    ![alt text](outputs/plots/xgboost.png)


    ## Step 3 : GitHub Repository Link
![alt text](image.png)

## Step 4 : Requirements.txt file
    

        scikit-learn
        matplotlib
        seaborn
        joblib
        pandas
        numpy
        xgboost
        PyYAML
        plotly

        graphviz

        xgboost

        streamlit==1.32.2
        altair==4.2.2