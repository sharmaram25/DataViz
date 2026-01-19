import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import streamlit as st
from scipy import stats

# --- Cached Helper Functions ---

@st.cache_data
def _test_normality(df, column):
    """
    Performs Agostino's K^2 test for normality.
    Returns (statistic, p_value, is_normal_0.05).
    """
    if column not in df.columns:
        return None, None, False
    data = df[column].dropna()
    if len(data) < 8: # Min samples for test
        return None, None, False
    
    # Sample if too large for performance
    if len(data) > 5000:
        data = data.sample(5000, random_state=42)
        
    try:
        k2, p = stats.normaltest(data)
        return k2, p, p > 0.05
    except Exception:
        return None, None, False

@st.cache_data
def _compute_target_correlation(df, target_col):
    """
    Computes correlation of all columns against a specific target.
    """
    numerics = df.select_dtypes(include=[np.number])
    if target_col not in numerics.columns:
        return pd.DataFrame()
    
    corrs = numerics.corrwith(numerics[target_col]).reset_index()
    corrs.columns = ["Feature", "Correlation"]
    corrs['AbsCorrelation'] = corrs['Correlation'].abs()
    return corrs.sort_values(by="AbsCorrelation", ascending=False)

@st.cache_data
def _compute_correlation_matrix(df, method='pearson'):
    numerics = df.select_dtypes(include=[np.number])
    if numerics.empty or numerics.shape[1] < 2:
        return None
    return numerics.corr(method=method)

@st.cache_data
def _detect_outliers_iqr(df, column):
    if column not in df.columns:
        return []
    series = df[column].dropna()
    if not np.issubdtype(series.dtype, np.number):
        return []
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    outliers = series[((series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR)))]
    return outliers.index.tolist()

@st.cache_data
def _run_pca_analysis(df):
    numerics = df.select_dtypes(include=[np.number]).dropna()
    
    # Sampling for PCA if data is too huge to keep it fast
    if len(numerics) > 5000:
        numerics = numerics.sample(5000, random_state=42)
    
    if numerics.shape[1] < 2 or numerics.shape[0] < 5:
        return None, "Not enough numerical data for PCA."
        
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numerics)
    
    pca = PCA(n_components=3)
    components = pca.fit_transform(scaled_data)
    
    pca_df = pd.DataFrame(data=components, columns=['PC1', 'PC2', 'PC3'])
    
    # Re-attach categorical info (needs alignment since we might have sampled)
    # For now, simplistic approach: just use indices if we hadn't sampled, 
    # but since we sampled, we can't easily map back without more logic.
    # So we'll just return the shape features.
    pca_df['Color_Dimension'] = "Sampled Data"
        
    return pca_df, pca.explained_variance_ratio_

class EDAAnalyzer:
    """
    Performs Exploratory Data Analysis computations, including advanced ML-lite techniques.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def get_descriptive_stats(self):
        # caching describe() might be overkill unless data is huge, but let's leave it dynamic for now
        desc = self.df.describe(include='all').T
        numerics = self.df.select_dtypes(include=[np.number])
        if not numerics.empty:
            skewness = numerics.skew()
            kurtosis = numerics.kurtosis()
            desc['skew'] = skewness
            desc['kurtosis'] = kurtosis
        return desc
        
    def compute_correlation_matrix(self, method='pearson'):
        return _compute_correlation_matrix(self.df, method)
        
    def detect_outliers_iqr(self, column):
        return _detect_outliers_iqr(self.df, column)

    def get_smart_insights(self):
        insights = []
        
        # Missing values
        missing_counts = self.df.isnull().sum()
        high_missing = missing_counts[missing_counts > len(self.df) * 0.5]
        for col, count in high_missing.items():
            insights.append(f"⚠️ Column **{col}** has {count/len(self.df):.1%} missing values. Consider excluding it.")
            
        # Skewness
        numerics = self.df.select_dtypes(include=[np.number])
        for col in numerics.columns:
            skew = numerics[col].skew()
            if skew > 2:
                insights.append(f"📈 Column **{col}** is heavily right-skewed (skew={skew:.2f}). Log-transform might help.")
            elif skew < -2:
                insights.append(f"📉 Column **{col}** is heavily left-skewed (skew={skew:.2f}).")
                
        # High Cardinality
        objects = self.df.select_dtypes(include=['object', 'category'])
        for col in objects.columns:
            if self.df[col].nunique() > len(self.df) * 0.9:
                insights.append(f"🆔 Column **{col}** looks like an ID (Unique ration > 90%).")
            elif self.df[col].nunique() == 1:
                insights.append(f"🛑 Column **{col}** is constant (single value). Dropping suggested.")
                
        return insights

    def run_pca_analysis(self):
        return _run_pca_analysis(self.df)

    def test_normality(self, column):
        return _test_normality(self.df, column)

    def compute_target_correlation(self, target_col):
        return _compute_target_correlation(self.df, target_col)
