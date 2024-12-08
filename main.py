import os
import json
from collections import Counter
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

data_dir = './test_data'

files = os.listdir(data_dir)

csv_files = [file for file in files if file.endswith('.csv')]
json_files = [file for file in files if file.endswith('.json')]

datasets = {}
categories = {}

for csv_file in csv_files:
    base_name = csv_file.split('.')[0]
    datasets[base_name] = pd.read_csv(os.path.join(data_dir, csv_file), encoding='latin1')

for json_file in json_files:
    base_name = json_file.split('.')[0]
    with open(os.path.join(data_dir, json_file), 'r') as f:
        categories[base_name] = json.load(f)

def map_categories(dataset_name):
    dataset = datasets[dataset_name]

    country_code = dataset_name.split("videos")[0].upper()  # Obtém o código do país (ex: CA, US)
    category_file = f"{country_code}_category_id"


    if category_file in categories:
        category_mapping = {
            int(item['id']): item['snippet']['title']
            for item in categories[category_file]['items']
        }

        dataset['category_title'] = dataset['category_id'].map(category_mapping)

        unmapped_ids = dataset[dataset['category_title'].isna()]['category_id'].unique()

        dataset['category_title'] = dataset['category_title'].fillna('Categoria Desconhecida')
    else:
        dataset['category_title'] = 'Categoria Desconhecida'

    return dataset

for dataset_name in datasets.keys():
    datasets[dataset_name] = map_categories(dataset_name)

combined_data = []
for dataset_name, dataset in datasets.items():
    dataset['region'] = dataset_name.replace("videos", "").upper()
    combined_data.append(dataset)

final_dataset = pd.concat(combined_data, ignore_index=True)

final_dataset.drop_duplicates(subset=['video_id'], inplace=True)
final_dataset.fillna({'likes': 0, 'dislikes': 0, 'comment_count': 0}, inplace=True)

final_dataset['engagement_rate'] = (final_dataset['likes'] + final_dataset['comment_count']) / final_dataset['views']
final_dataset['like_dislike_ratio'] = final_dataset['likes'] / (final_dataset['dislikes'] + 1)

print("Colunas do dataset:")
print(final_dataset.columns)

X = final_dataset.drop(columns=['category_title'])  # Ajuste o nome da coluna alvo
y = final_dataset['category_title']

if y.dtypes == 'object':
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

for column in X.select_dtypes(include=['object']).columns:
    label_encoder = LabelEncoder()
    X[column] = label_encoder.fit_transform(X[column])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Distribuição original das classes no treino:", Counter(y_train))

X_train = X_train.reset_index(drop=True)
y_train_series = pd.Series(y_train).reset_index(drop=True)

min_samples = 20
filtered_indices = y_train_series.isin([cls for cls, count in Counter(y_train_series).items() if count >= min_samples])

X_train_filtered = X_train[filtered_indices]
y_train_filtered = y_train_series[filtered_indices]

print("Distribuição após remoção de classes pequenas:", Counter(y_train_filtered))

smote = SMOTE(random_state=42, k_neighbors=2)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_filtered, y_train_filtered)

print("Distribuição após SMOTE:", Counter(y_train_balanced))

def evaluate_model(model, name):
    skf = StratifiedKFold(n_splits=5)
    cv_scores = cross_val_score(model, X_train_balanced, y_train_balanced, cv=skf, scoring='accuracy')

    model.fit(X_train_balanced, y_train_balanced)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"--- {name} ---")
    print(f"Cross-Validation Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Classification Report:\n{report}\n")

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_balanced, y_train_balanced)
model_path = './random_forest_model.joblib'
joblib.dump(rf, model_path)

print(f"Model saved to {model_path}")

evaluate_model(rf, "Random Forest")
