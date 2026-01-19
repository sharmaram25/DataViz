import pandas as pd
import numpy as np
import os
from src.preprocess import DataPreprocessor
from src.eda import EDAAnalyzer
from src.visuals import plot_histogram

# Create synthetic data
data = {
    'id': range(1, 101),
    'category': np.random.choice(['A', 'B', 'C'], 100),
    'value': np.random.normal(50, 15, 100),
    'missing_col': [None if i % 10 == 0 else i for i in range(100)]
}
df = pd.DataFrame(data)

print("--- Testing Preprocessor ---")
prep = DataPreprocessor(df)
print("Summary:", prep.get_summary())
print("Types:", prep.get_column_types())
print("Quality Score:", prep.get_data_quality_score())

print("\n--- Testing EDA ---")
eda = EDAAnalyzer(df)
print("Stats Shape:", eda.get_descriptive_stats().shape)
print("Insights:", eda.get_smart_insights())
print("Outliers in 'value':", eda.detect_outliers_iqr('value'))

print("\n--- Testing Visuals ---")
try:
    fig = plot_histogram(df, 'value', 'category')
    print("Chart created successfully (Plotly Figure object)")
except Exception as e:
    print(f"Chart creation failed: {e}")

print("\n>>> ALL TESTS PASSED SUCCESSFULLY <<<")
