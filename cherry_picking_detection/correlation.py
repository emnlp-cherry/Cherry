from scipy import stats
import seaborn as sns
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def visualize_pearson_correlation(x,y, name1,name2):
    df = pd.DataFrame(list(zip(x, y)),columns =[name1, name2])
    sns.scatterplot(x=name1, y=name2, data=df)


def calculate_correlation(x,y, name1,name2):
    df = pd.DataFrame(list(zip(x, y)), columns=[name1, name2])
    pearson_r, p_p_value = stats.pearsonr(df[name1], df[name2])
    spearman_r, s_p_value = stats.spearmanr(df[name1], df[name2])
    print("Pearson's coefficient (r) = "+str(pearson_r),"P-value = "+str(p_p_value))
    print("Spearman's coefficient (r) = "+str(spearman_r),"P-value = "+str(s_p_value))
    return pearson_r



