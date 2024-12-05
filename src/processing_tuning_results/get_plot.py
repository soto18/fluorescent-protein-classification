import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, "../")
import pandas as pd
from src.processing_explored_results.ploting_figures import MakePlots

class PerformanceMergerAndPlotter:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.df_data = None
        self.df_processed = None

    def load_data(self, file_name):
        self.df_data = pd.read_csv(f"{self.input_path}/{file_name}")
        print(f"Data loaded from {self.input_path}/{file_name}")

    def merge_documents(self):
        if self.df_data is None:
            raise ValueError("No data loaded. Use 'load_data' first.")

        # Ajustar los nombres de las columnas de entrenamiento
        df_results_train = self.df_data[['encoder', 'algorithm', 'train_f1', 'train_recall', 'train_precision', 'train_accuracy', "random_seed"]]
        df_results_train.columns = ["Encoder", "Algorithm", "F1", "Recall", "Precision", "Accuracy", "random_seed"]

        # Ajustar los nombres de las columnas de validación
        df_results_test = self.df_data[['encoder', 'algorithm', 'test_accuracy', 'test_precision', 'test_recall', 'test_f1', "random_seed"]]
        df_results_test.columns = ["Encoder", "Algorithm", "Accuracy", "Precision", "Recall", "F1", "random_seed"]

        df_results_train["Stage"] = "Training"
        df_results_test["Stage"] = "Test"

        # Concatenar y procesar el DataFrame
        df_process = pd.concat([df_results_train, df_results_test], axis=0)
        df_process.reset_index(inplace=True)

        # Clasificar el tipo de encoder
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
        self.df_processed = df_process

    def save_processed_data(self, file_name):
        if self.df_processed is None:
            raise ValueError("No processed data to save. Use 'merge_documents' first.")

        self.df_processed.to_csv(f"{self.output_path}/{file_name}", index=False)
        print(f"Processed data saved to {self.output_path}/{file_name}")

    def plot_data(self):
        if self.df_processed is None:
            raise ValueError("No processed data available for plotting. Use 'merge_documents' first.")

        print("Plotting data...")
        make_plots = MakePlots(dataset=self.df_processed, path_export=self.output_path, hue="Stage")
        make_plots.plot_by_algorithm()  # Ahora podrá incluir `Algorithm` en los gráficos
        make_plots.plot_by_encoder()
        make_plots.plot_by_type_encoder()
        make_plots.plot_filter_by_nlp()
        print("Plots generated.")

    def run(self, input_file, output_file):
        self.load_data(input_file)
        self.merge_documents()
        self.save_processed_data(output_file)
        self.plot_data()
