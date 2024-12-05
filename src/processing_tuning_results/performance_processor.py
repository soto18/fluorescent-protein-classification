import pandas as pd
import os

class PerformanceProcessor:
    def __init__(self, path_results="../../results/performance", path_export="../../results/summary_exploring/"):
        self.path_results = path_results
        self.path_export = path_export
        self.list_type_encoders = ['embedding', 'FFT', 'physicochemical_properties']
        self.list_df = []

    def process_encoder(self, type_encoder):
        print("Processing encoder:", type_encoder)
        encoder_path = os.path.join(self.path_results, type_encoder)
        list_encoder = os.listdir(encoder_path)

        for encoder in list_encoder:
            list_files = os.listdir(os.path.join(encoder_path, encoder))

            for doc_file in list_files:
                if doc_file.endswith(".csv"):
                    df_data = pd.read_csv(os.path.join(encoder_path, encoder, doc_file))
                    df_data["encoder"] = f"{encoder}_{type_encoder}"
                    df_data["algorithm"] = doc_file.split("_")[2].split(".")[0]
                    df_data["random_seed"] = 42
                    self.list_df.append(df_data)

    def process_all_encoders(self):
        os.makedirs(self.path_export, exist_ok=True)
        for type_encoder in self.list_type_encoders:
            self.process_encoder(type_encoder)

    def save_concatenated_data(self, file_name="explored_data.csv"):
        if self.list_df:
            df_explored = pd.concat(self.list_df, axis=0)
            export_path = os.path.join(self.path_export, file_name)
            df_explored.to_csv(export_path, index=False)
            print(f"Data exported to: {export_path}")

    def run(self):
        self.process_all_encoders()
        self.save_concatenated_data()
