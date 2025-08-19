# Filename: 3_explain_model.py
import h2o
import shap
import pandas as pd
import matplotlib.pyplot as plt
h2o.init()
# Load the saved model (replace with your actual model ID from the leaderboard)
model_id = "StackedEnsemble_AllModels_1_AutoML_1_..."
best_model = h2o.get_model(model_id)
# Load test data for explanation
test_df = pd.read_csv("dga_dataset_train.csv")
X_test = test_df[['length', 'entropy']]
def predict_wrapper(data):
    h2o_df = h2o.H2OFrame(pd.DataFrame(data, columns=X_test.columns))
    predictions = best_model.predict(h2o_df)
    return predictions.as_data_frame()['dga']
explainer = shap.KernelExplainer(predict_wrapper, X_test.head(50))
shap_values = explainer.shap_values(X_test.head(50))
print("Displaying SHAP Summary Plot (Global Explanation)...")
shap.summary_plot(shap_values, X_test.head(50), show=False)
plt.savefig("shap_summary.png")
plt.close()
print("Displaying SHAP Force Plot (Local Explanation for first instance)...")
shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:], show=False, matplotlib=True)
plt.savefig("shap_force.png")
plt.close()
h2o.shutdown()