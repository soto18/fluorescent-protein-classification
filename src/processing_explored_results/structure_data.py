import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from ploting_figures import MakePlots

def merge_documents(df_data=None):
    df_results_train = df_data[['encoder', 'algorithm', 'F1_cv', 'recall_cv', 'precision_cv', 'accuracy_cv', "random_seed"]]
    df_results_train.columns = ["Encoder", "Algorithm", "F1", "Recall", "Precision", "Accuracy", "random_seed"]

    df_results_test = df_data[['encoder', 'algorithm', 'accuracy_val', 'precision_val', 'recall_val', 'f1_val', "random_seed"]]
    df_results_test.columns = ["Encoder", "Algorithm", "Accuracy", "Precision", "Recall", "F1", "random_seed"]

    df_results_train["Stage"] = "Training"
    df_results_test["Stage"] = "Validation"

    df_process = pd.concat([df_results_train, df_results_test], axis=0)

    df_process.reset_index(inplace=True)

    type_encoder = []

    for index in df_process.index:
        name_encoder = df_process["Encoder"][index]

        if "FFT" in name_encoder:
            type_encoder.append("FFT-Based")
        elif "physicochemical_properties" in name_encoder:
            type_encoder.append("PHY-Based")
        elif "onehot" in name_encoder:
            type_encoder.append("One-Hot-Based")
        else:
            type_encoder.append("NLP-Based")

    df_process["Type-Encoder"] = type_encoder
    return df_process


df_data = pd.read_csv("../../results/summary_exploring/explored_data.csv")
path_results = "../../results/summary_exploring/"
df_process = merge_documents(df_data=df_data)
df_process.to_csv(f"{path_results}processed_performances.csv", index=False)

print("Plotting data")
make_plots = MakePlots(dataset=df_process, path_export=path_results, hue="Stage")
make_plots.plot_by_algorithm()
make_plots.plot_by_encoder()
make_plots.plot_by_type_encoder()
make_plots.plot_filter_by_nlp()