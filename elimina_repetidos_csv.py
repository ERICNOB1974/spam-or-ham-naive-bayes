import pandas as pd

dataframe = pd.read_csv("./datasets/ds_llm.csv")  
#df = pd.read_csv("./datasets/ds_huggingface.csv") 

dataframe_sin_duplicados = dataframe.drop_duplicates()

dataframe_sin_duplicados.to_csv("./datasets/ds_llm_limpio.csv", index=False)