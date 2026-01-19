import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype

class DataPreprocessor:
    """
    Handles data metadata extraction, type classification, basic quality checks,
    and filtering operations.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def get_summary(self):
        """
        Returns basic dataset metadata.
        """
        return {
            "rows": self.df.shape[0],
            "cols": self.df.shape[1],
            "memory_usage": f"{self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
            "duplicates": self.df.duplicated().sum(),
            "missing_cells": self.df.isnull().sum().sum(),
            "missing_percent": (self.df.isnull().sum().sum() / self.df.size) * 100
        }

    def get_column_types(self):
        """
        Classifies columns into Numerical, Categorical, Datetime, and Text.
        Returns a dictionary with lists of column names.
        """
        numerics = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Try to distinguish actual text from categorical (low cardinality text)
        objects = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        categorical = []
        text = []
        
        for col in objects:
            try:
                unique_count = self.df[col].nunique()
                # Heuristic: if unique count is low (< 50 or < 5% of rows), treat as categorical
                if unique_count < 50 or unique_count < len(self.df) * 0.05:
                    categorical.append(col)
                else:
                    text.append(col)
            except:
                text.append(col)
                
        # Datetime detection (naive but effective for clean data)
        datetimes = self.df.select_dtypes(include=['datetime', 'timedelta']).columns.tolist()
            
        return {
            "numerical": numerics,
            "categorical": categorical,
            "text": text,
            "datetime": datetimes
        }

    def get_data_quality_score(self):
        """
        Computes a simple data quality score (0-100) based on missingness and duplicates.
        """
        total_score = 100
        
        # Penalize for missing values
        missing_pct = (self.df.isnull().sum().sum() / self.df.size) * 100
        total_score -= missing_pct * 0.5  # 10% missing = -5 points
        
        # Penalize for duplicates (max 20 points deduction)
        dupe_pct = (self.df.duplicated().sum() / len(self.df)) * 100
        total_score -= min(dupe_pct, 20)
        
        return max(0, round(total_score))

    def filter_data(self, filters):
        """
        Applies a dictionary of filters to the dataframe.
        filters = {'col_name': (min, max)} or {'col_name': [categories]}
        """
        df_filtered = self.df.copy()
        for col, value in filters.items():
            if col not in df_filtered.columns:
                continue
                
            if is_numeric_dtype(df_filtered[col]):
                min_val, max_val = value
                df_filtered = df_filtered[(df_filtered[col] >= min_val) & (df_filtered[col] <= max_val)]
            
            elif is_datetime64_any_dtype(df_filtered[col]):
                min_val, max_val = value
                df_filtered = df_filtered[(df_filtered[col] >= min_val) & (df_filtered[col] <= max_val)]
                
            else:
                # Categorical / Text
                if value: # if list is not empty
                    df_filtered = df_filtered[df_filtered[col].isin(value)]
                    
        return df_filtered
