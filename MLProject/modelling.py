import pandas as pd
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import os

def train_model():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="dataset_preprocessing/titanic_processed.csv")
    args = parser.parse_args()

    mlflow.autolog()
    
    if not os.path.exists(args.data_path):
        return
        
    df = pd.read_csv(args.data_path)
    
    X = df.drop(columns=['Survived'])
    y = df['Survived']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    with mlflow.start_run():
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        print(acc)
        
if __name__ == "__main__":
    train_model()
