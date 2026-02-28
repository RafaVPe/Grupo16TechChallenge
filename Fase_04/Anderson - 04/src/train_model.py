from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline 

from xgboost import XGBClassifier

import joblib
import pandas as pd


def main(): 

    # Carregando os dados
    df_obesity1 = pd.read_csv("../data/raw/Obesity.csv")
    

    cols_to_round = ['FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    
    df_obesity1[cols_to_round] = (
        df_obesity1[cols_to_round].round().astype(int)
    )

    # Preparando X e y sem tranformação manual
    X = df_obesity1.drop("Obesity", axis=1)
    y = df_obesity1["Obesity"]

    #Codificar target (necessário para o XGBoost)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    
    #Identificando as colunas
    numeric_features = [
    "Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"    
        ]

    categ_features = [ 
    "Gender" ,"family_history", "FAVC", "CAEC", "SMOKE", "SCC", "CALC", "MTRANS"
        ]

    # Criando o pré processador
    preprocessor = ColumnTransformer(
        transformers=[
            ("Num", StandardScaler(), numeric_features),
            ("Cat", OneHotEncoder(handle_unknown="ignore"), categ_features)
        ]
    )

    # Modelo final 
    xgb_model = Pipeline(
    steps = [
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(eval_metric="mlogloss", random_state=42))
        ]
    )

    # Treinando com todos os dados
    xgb_model.fit(X, y_encoded)

    joblib.dump(xgb_model, "../models/modelo_obesidade.pkl")
    joblib.dump(le,"../models/label_encoder.pkl")

    print("Modelo treinado e salvo com sucesso") 


if __name__ == "__main__":
    main()

