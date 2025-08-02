import pandas as pd
import deepchem as dc

#Standardized parameters
mean = 0.487117073991484184070088758745
std = 0.049470465217897587051343322173

model = dc.models.GraphConvModel(1, graph_conv_layers=[30, 30],
              mode="regression",
              batch_normalize=False,
              model_dir=r"model")
model.restore()

# Validation set file path
DATASET_FILE = r'Validation_Set.csv'

# Featurization
featurizer = dc.feat.ConvMolFeaturizer()
loader = dc.data.CSVLoader(tasks=[], feature_field="SMILES", featurizer=featurizer)
dataset = loader.create_dataset(DATASET_FILE, shard_size=10000)
print("\nLoad data successfully！\n")

# Make predictions
test_pred = model.predict(dataset)
test_pred = 10**((test_pred * std) + mean)
df = pd.read_csv(DATASET_FILE)
molecule_names = df['SMILES']  
results_df = pd.DataFrame({
    'SMILES': molecule_names,  
    'pre': test_pred.flatten()
})

# Write to a new CSV file
output_file = r'Prediction.csv'
results_df.to_csv(output_file, index=False)
print(f"Predictions successfully saved to {output_file}")
