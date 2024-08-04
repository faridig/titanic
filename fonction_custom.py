import pandas as pd

#fonction transformer custom : preprocess de cabin
def extract_cabin_letter(df):
    return pd.DataFrame(df.str[0])