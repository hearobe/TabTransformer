import pandas as pd
import json

from pytorch_widedeep import Trainer
from pytorch_widedeep.models import TabTransformer, WideDeep
from pytorch_widedeep.preprocessing import TabPreprocessor
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

# Load dataframes from P files created in DataCleaning&FeatureEngineering.py
def load_dataset():

    train = pd.read_pickle("app/data_train.p")
    valid = pd.read_pickle("app/data_val.p")
    test = pd.read_pickle("app/data_test.p")

    train = pd.concat([train, valid], ignore_index=True)

    return train, test

# Format data for TabTransformer model
def prepare_data():

    train, test = load_dataset()

    cat_embed_cols = ["gender",'SeniorCitizen','Partner','Dependents','PhoneService','MultipleLines','InternetService',
                      'OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies',
                     'PaperlessBilling','Contract','PaymentMethod','bin_TotalCharges','bin_MonthlyCharges','bin_tenure']
    con_embed_cols = [c for c in train.columns if c != "Default" and c not in cat_embed_cols]


    prepare_tab = TabPreprocessor(embed_cols=cat_embed_cols, continuous_cols = con_embed_cols, for_transformer=True, scale=False)
    X_train = prepare_tab.fit_transform(train)
    y_train = train.Default.values
    X_test = prepare_tab.transform(test)
    y_test = test.Default.values

    return prepare_tab, X_train, X_test, y_train, y_test

# Set TabTransformer model using return values from prepare_data
def set_model(prepare_tab, depth):
    # if no depth given, set to the depth recommended by paper
    if(depth == 0):
        depth = 6

    deeptabular = TabTransformer(
        column_idx=prepare_tab.column_idx,
        embed_input=prepare_tab.embeddings_input,
        n_blocks=6,
        n_heads=8
    )

    model = WideDeep(deeptabular=deeptabular)

    return model
    
# Runs and returns results of predictive model
def run_experiment_and_save(
    model,
    X_train,
    X_test,
    y_train,
    y_test
):

    trainer = Trainer(
        model,
        objective="binary"
    )

    trainer.fit(
        X_train={"X_tab": X_train, "target": y_train},
        X_val={"X_tab": X_test, "target": y_test}
    )

    y_pred = trainer.predict(X_tab=X_test)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    results_dict = {
        'Accuracy': acc,
        'F1': f1,
        'ROC_AUC': auc,
        "True_Negative":int(tn),
        "False_Positive":int(fp),
        'False_Negative':int(fn),
        'True_Positive':int(tp)
    }
    results = json.dumps(results_dict)
    return results
