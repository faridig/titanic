import pandas as pd

#fonction transformer custom : preprocess de cabin
def extract_cabin_letter(serie):
    return pd.DataFrame(serie.str[0])