import time
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.eda import _compute_correlation_matrix, _run_pca_analysis, _detect_outliers_iqr
from src.visuals import sample_data

def generate_large_dataset(rows=500000):
    print(f"Generating synthetic dataset with {rows} rows...")
    df = pd.DataFrame({
        'id': range(rows),
        'category': np.random.choice(['A', 'B', 'C', 'D'], rows),
        'value1': np.random.randn(rows),
        'value2': np.random.rand(rows) * 100,
        'value3': np.random.randint(0, 1000, rows),
        'date': pd.date_range('2020-01-01', periods=rows, freq='min')
    })
    return df

def run_stress_test():
    start_total = time.time()
    
    # 1. Data Generation
    t0 = time.time()
    df = generate_large_dataset(100000) # Start with 100k for quick test
    print(f"✅ Data Gen: {time.time() - t0:.4f}s")
    
    # 2. visual sampling
    t0 = time.time()
    sampled = sample_data(df)
    print(f"✅ Visual Sampling (100k -> {len(sampled)}): {time.time() - t0:.4f}s")
    assert len(sampled) <= 5000
    
    # 3. PCA (Heavy)
    t0 = time.time()
    # Emulate the logic inside eda.py
    # We need to make sure we're calling the cached function logic
    # In a real run, streamlit caches it. Here we just measure execution time.
    pca_res, var = _run_pca_analysis(df) 
    print(f"✅ PCA Analysis: {time.time() - t0:.4f}s")
    
    # 4. Correlation
    t0 = time.time()
    corr = _compute_correlation_matrix(df)
    print(f"✅ Correlation Matrix: {time.time() - t0:.4f}s")
    
    print(f"\n🚀 Total Stress Test Time: {time.time() - start_total:.4f}s")
    
if __name__ == "__main__":
    run_stress_test()
