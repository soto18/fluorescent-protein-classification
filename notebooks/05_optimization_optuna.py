import pandas as pd
import sys
sys.path.insert(0, "../")
from src.training_models.optimization_models import OptimizationProcess

base_path = "../results/encoders/"
algorithms = [
    "RandomForestClassifier", "GradientBoostingClassifier", "ExtraTreesClassifier", 
    "AdaBoostClassifier", "KNeighborsClassifier", "DecisionTreeClassifier", "SVC"]
encoders_config = {
    "FFT": ["CRAJ730102"],
    "embedding": ["bepler", "esm1b", "esme"],
    "physicochemical_properties": ["CRAJ730102"]
}

for type_encoder, encoders in encoders_config.items():
    for encoder in encoders:
        df_data = pd.read_csv(f"{base_path}/{type_encoder}/{encoder}/coded_dataset.csv")
        for algorithm in algorithms:
            print(f"Running optimization for {type_encoder}: {encoder}, Algorithm: {algorithm}")
            
            model_instance = OptimizationProcess(df_data=df_data)
            try:
                model_instance.run_optimization(
                    type_encoder=type_encoder, 
                    encoder=encoder,
                    algorithm=algorithm
                )
            except Exception as e:
                print(f"Error during optimization for {type_encoder} - {encoder} - {algorithm}: {e}")