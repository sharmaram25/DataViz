"""
EDA Utilities - Advanced data analysis and visualization functions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class DataAnalyzer:
    """Comprehensive data analysis class"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
        # Improved column type detection
        self._detect_column_types()
    
    def _detect_column_types(self):
        """Enhanced column type detection"""
        # Initialize lists
        self.numeric_cols = []
        self.categorical_cols = []
        self.datetime_cols = []
        self.boolean_cols = []
        
        for col in self.df.columns:
            col_data = self.df[col]
            
            # Skip empty columns
            if col_data.isna().all():
                continue
                
            # Check for datetime columns
            if col_data.dtype.name.startswith('datetime'):
                self.datetime_cols.append(col)
                continue
            
            # Check for boolean columns
            if col_data.dtype == 'bool' or set(col_data.dropna().unique()).issubset({True, False, 1, 0}):
                self.boolean_cols.append(col)
                continue
            
            # Try to convert to numeric
            try:
                # Remove non-numeric characters and try conversion
                if col_data.dtype == 'object':
                    # Check if it looks like numeric data
                    sample_values = col_data.dropna().astype(str).str.replace(r'[,$%]', '', regex=True)
                    pd.to_numeric(sample_values.head(100), errors='raise')
                    
                    # If successful, convert the column
                    cleaned_col = col_data.astype(str).str.replace(r'[,$%]', '', regex=True)
                    self.df[col] = pd.to_numeric(cleaned_col, errors='coerce')
                    self.numeric_cols.append(col)
                elif col_data.dtype in ['int64', 'int32', 'float64', 'float32']:
                    self.numeric_cols.append(col)
                else:
                    # Check if it's actually numeric
                    if pd.api.types.is_numeric_dtype(col_data):
                        self.numeric_cols.append(col)
                    else:
                        self.categorical_cols.append(col)
            except:
                # If conversion fails, treat as categorical
                # But first check if it's a small set of unique values that might be categorical
                unique_count = col_data.nunique()
                total_count = len(col_data.dropna())
                
                if unique_count <= max(20, total_count * 0.05):  # Less than 5% unique or max 20 unique values
                    self.categorical_cols.append(col)
                else:
                    # High cardinality text data
                    self.categorical_cols.append(col)
    
    def show_dataset_overview(self):
        """Display comprehensive dataset overview"""
        st.markdown("## 📊 Dataset Overview")
        
        # Basic info metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📏 Total Rows",
                value=f"{self.df.shape[0]:,}",
                help="Number of observations in the dataset"
            )
        
        with col2:
            st.metric(
                label="📊 Total Columns", 
                value=self.df.shape[1],
                help="Number of features/variables"
            )
        
        with col3:
            memory_usage = self.df.memory_usage(deep=True).sum()
            memory_mb = memory_usage / (1024 * 1024)
            st.metric(
                label="💾 Memory Usage",
                value=f"{memory_mb:.2f} MB",
                help="Total memory consumption"
            )
        
        with col4:
            missing_percentage = (self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1])) * 100
            st.metric(
                label="❓ Missing Data",
                value=f"{missing_percentage:.1f}%",
                help="Percentage of missing values"
            )
        
        # Feature type classification
        st.markdown("### 🏷️ Feature Type Classification")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Feature types pie chart
            feature_counts = {
                'Numerical': len(self.numeric_cols),
                'Categorical': len(self.categorical_cols),
                'DateTime': len(self.datetime_cols),
                'Boolean': len(self.boolean_cols)
            }
            
            # Remove zero counts
            feature_counts = {k: v for k, v in feature_counts.items() if v > 0}
            
            if feature_counts:
                fig = px.pie(
                    values=list(feature_counts.values()),
                    names=list(feature_counts.keys()),
                    title="Feature Type Distribution",
                    color_discrete_sequence=['#3498db', '#e74c3c', '#f39c12', '#27ae60']
                )
                fig.update_layout(
                    height=400,
                    title_font_size=16,
                    font=dict(size=12)
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Data types table
            dtype_df = pd.DataFrame({
                'Column': self.df.columns,
                'Data Type': self.df.dtypes.astype(str),
                'Non-Null Count': self.df.count(),
                'Null Count': self.df.isnull().sum(),
                'Null %': (self.df.isnull().sum() / len(self.df) * 100).round(2)
            })
            
            st.markdown("**Column Information:**")
            st.dataframe(
                dtype_df,
                use_container_width=True,
                height=400
            )
        
        # Data quality insights
        st.markdown("### 🔍 Data Quality Insights")
        
        insights_col1, insights_col2, insights_col3 = st.columns(3)
        
        with insights_col1:
            # Constant columns
            constant_cols = [col for col in self.df.columns if self.df[col].nunique() <= 1]
            if constant_cols:
                st.warning(f"**⚠️ Constant Columns ({len(constant_cols)}):**\n" + 
                          "\n".join([f"• {col}" for col in constant_cols]))
            else:
                st.success("✅ No constant columns found")
        
        with insights_col2:
            # Duplicate rows
            duplicate_count = self.df.duplicated().sum()
            if duplicate_count > 0:
                st.warning(f"**🔁 Duplicate Rows:** {duplicate_count:,}")
            else:
                st.success("✅ No duplicate rows found")
        
        with insights_col3:
            # High correlation pairs
            if len(self.numeric_cols) > 1:
                corr_matrix = self.df[self.numeric_cols].corr()
                # Get upper triangle and find high correlations
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
                high_corr = np.where((np.abs(corr_matrix) > 0.95) & mask)
                
                if len(high_corr[0]) > 0:
                    high_corr_pairs = []
                    for i, j in zip(high_corr[0], high_corr[1]):
                        col1, col2 = corr_matrix.index[i], corr_matrix.columns[j]
                        corr_val = corr_matrix.iloc[i, j]
                        high_corr_pairs.append(f"{col1} ↔ {col2} ({corr_val:.3f})")
                    
                    st.warning(f"**🔗 High Correlations (>0.95):**\n" + 
                              "\n".join([f"• {pair}" for pair in high_corr_pairs[:5]]))
                else:
                    st.success("✅ No high correlations found")
            else:
                st.info("ℹ️ Need 2+ numeric columns for correlation analysis")
    
    def show_descriptive_statistics(self):
        """Display detailed descriptive statistics"""
        st.markdown("## 📉 Descriptive Statistics")
        
        # Numerical statistics
        if self.numeric_cols:
            st.markdown("### 🔢 Numerical Features Analysis")
            
            # Enhanced numerical statistics
            numeric_df = self.df[self.numeric_cols]
            
            # Basic statistics
            desc_stats = numeric_df.describe()
            
            # Add skewness and kurtosis
            skewness = numeric_df.skew()
            kurtosis = numeric_df.kurtosis()
            
            # Combine statistics
            enhanced_stats = desc_stats.copy()
            enhanced_stats.loc['skewness'] = skewness
            enhanced_stats.loc['kurtosis'] = kurtosis
            enhanced_stats.loc['iqr'] = enhanced_stats.loc['75%'] - enhanced_stats.loc['25%']
            
            st.dataframe(
                enhanced_stats.round(3),
                use_container_width=True
            )
            
            # Distribution insights
            st.markdown("### 📊 Distribution Insights")
            
            if len(self.numeric_cols) > 0:
                selected_col = st.selectbox(
                    "Select column for detailed analysis:",
                    self.numeric_cols,
                    key="numeric_detail"
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Distribution metrics
                    col_data = self.df[selected_col].dropna()
                    
                    metrics_data = {
                        'Mean': col_data.mean(),
                        'Median': col_data.median(),
                        'Mode': col_data.mode().iloc[0] if not col_data.mode().empty else 'N/A',
                        'Std Dev': col_data.std(),
                        'Variance': col_data.var(),
                        'Skewness': col_data.skew(),
                        'Kurtosis': col_data.kurtosis(),
                        'Min': col_data.min(),
                        'Max': col_data.max(),
                        'IQR': col_data.quantile(0.75) - col_data.quantile(0.25)
                    }
                    
                    metrics_df = pd.DataFrame(
                        list(metrics_data.items()),
                        columns=['Metric', 'Value']
                    )
                    
                    st.dataframe(
                        metrics_df,
                        use_container_width=True,
                        hide_index=True
                    )
                
                with col2:
                    # Distribution interpretation
                    skew_val = col_data.skew()
                    kurt_val = col_data.kurtosis()
                    
                    # Interpretation
                    skew_interpretation = (
                        "Right-skewed (positive)" if skew_val > 0.5 else
                        "Left-skewed (negative)" if skew_val < -0.5 else
                        "Approximately symmetric"
                    )
                    
                    kurt_interpretation = (
                        "Leptokurtic (heavy-tailed)" if kurt_val > 3 else
                        "Platykurtic (light-tailed)" if kurt_val < 3 else
                        "Mesokurtic (normal-like)"
                    )
                    
                    st.markdown(f"""
                    **Distribution Analysis:**
                    
                    📐 **Skewness:** {skew_val:.3f}  
                    ➡️ {skew_interpretation}
                    
                    📏 **Kurtosis:** {kurt_val:.3f}  
                    ➡️ {kurt_interpretation}
                    
                    🎯 **Outlier Detection:**  
                    Using IQR method
                    """)
                    
                    # Outlier detection
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                    
                    if len(outliers) > 0:
                        st.warning(f"⚠️ **{len(outliers)} outliers detected** ({len(outliers)/len(col_data)*100:.1f}%)")
                    else:
                        st.success("✅ No outliers detected")
        
        # Categorical statistics
        if self.categorical_cols:
            st.markdown("### 🏷️ Categorical Features Analysis")
            
            cat_stats_list = []
            
            for col in self.categorical_cols:
                col_data = self.df[col]
                stats_dict = {
                    'Column': col,
                    'Count': col_data.count(),
                    'Unique': col_data.nunique(),
                    'Most Frequent': col_data.mode().iloc[0] if not col_data.mode().empty else 'N/A',
                    'Frequency': col_data.value_counts().iloc[0] if len(col_data.value_counts()) > 0 else 0,
                    'Cardinality': col_data.nunique() / len(col_data) if len(col_data) > 0 else 0
                }
                cat_stats_list.append(stats_dict)
            
            cat_stats_df = pd.DataFrame(cat_stats_list)
            st.dataframe(
                cat_stats_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Detailed categorical analysis
            if len(self.categorical_cols) > 0:
                selected_cat = st.selectbox(
                    "Select categorical column for detailed analysis:",
                    self.categorical_cols,
                    key="cat_detail"
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Value counts
                    value_counts = self.df[selected_cat].value_counts().head(10)
                    
                    st.markdown(f"**Top 10 Values in '{selected_cat}':**")
                    st.dataframe(
                        pd.DataFrame({
                            'Value': value_counts.index,
                            'Count': value_counts.values,
                            'Percentage': (value_counts.values / len(self.df) * 100).round(2)
                        }),
                        use_container_width=True,
                        hide_index=True
                    )
                
                with col2:
                    # Cardinality analysis
                    unique_count = self.df[selected_cat].nunique()
                    total_count = len(self.df)
                    cardinality_ratio = unique_count / total_count
                    
                    cardinality_type = (
                        "High cardinality" if cardinality_ratio > 0.9 else
                        "Medium cardinality" if cardinality_ratio > 0.1 else
                        "Low cardinality"
                    )
                    
                    st.markdown(f"""
                    **Cardinality Analysis:**
                    
                    🔢 **Unique Values:** {unique_count:,}  
                    📊 **Total Values:** {total_count:,}  
                    📈 **Cardinality Ratio:** {cardinality_ratio:.3f}  
                    🏷️ **Type:** {cardinality_type}
                    """)
                    
                    if cardinality_ratio > 0.9:
                        st.warning("⚠️ High cardinality may indicate need for grouping")
                    elif cardinality_ratio < 0.01:
                        st.info("ℹ️ Very low cardinality - consider target encoding")
    
    def show_missing_values_analysis(self):
        """Comprehensive missing values analysis with smart suggestions"""
        st.markdown("## ❓ Missing Values Analysis")
        
        missing_counts = self.df.isnull().sum()
        missing_percentages = (missing_counts / len(self.df)) * 100
        
        # Check if there are any missing values
        total_missing = missing_counts.sum()
        
        if total_missing == 0:
            st.success("🎉 **Congratulations! Your data is completely clean!**")
            st.balloons()
            st.markdown("""
            <div style='background: linear-gradient(135deg, #27ae60, #2ecc71); 
                        color: white; padding: 2rem; border-radius: 10px; text-align: center;'>
                <h3>✨ No missing values detected!</h3>
                <p>Your dataset is ready for analysis without any data cleaning steps.</p>
            </div>
            """, unsafe_allow_html=True)
            return
        
        # Missing values summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="Total Missing Values",
                value=f"{total_missing:,}",
                delta=f"{(total_missing / (len(self.df) * len(self.df.columns)) * 100):.1f}% of total data"
            )
        
        with col2:
            affected_columns = (missing_counts > 0).sum()
            st.metric(
                label="Affected Columns",
                value=f"{affected_columns}",
                delta=f"{(affected_columns / len(self.df.columns) * 100):.1f}% of columns"
            )
        
        # Missing values table
        missing_df = pd.DataFrame({
            'Column': missing_counts.index,
            'Missing Count': missing_counts.values,
            'Missing %': missing_percentages.round(2),
            'Data Type': self.df.dtypes.astype(str)
        })
        
        # Filter to show only columns with missing values
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False)
        
        if not missing_df.empty:
            st.markdown("### 📊 Missing Values Summary")
            st.dataframe(
                missing_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Missing values heatmap
                st.markdown("**Missing Values Heatmap**")
                
                # Create binary missing values dataframe
                missing_binary = self.df.isnull().astype(int)
                
                if missing_binary.sum().sum() > 0:
                    fig = px.imshow(
                        missing_binary.T,
                        color_continuous_scale=['#ecf0f1', '#e74c3c'],
                        title="Missing Values Pattern",
                        labels=dict(x="Rows", y="Columns", color="Missing")
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Missing percentage bar chart
                st.markdown("**Missing Percentage by Column**")
                
                if not missing_df.empty:
                    fig = px.bar(
                        missing_df.head(10),
                        x='Missing %',
                        y='Column',
                        orientation='h',
                        title="Top 10 Columns with Missing Values",
                        color='Missing %',
                        color_continuous_scale='Reds'
                    )
                    fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
            
            # Smart imputation suggestions
            st.markdown("### 🤖 Smart Imputation Suggestions")
            
            suggestions_data = []
            
            for col in missing_df['Column']:
                missing_pct = missing_percentages[col]
                col_data = self.df[col]
                data_type = str(self.df[col].dtype)
                
                # Generate suggestions based on data type and missing percentage
                if missing_pct > 70:
                    suggestion = "🗑️ Consider dropping this column (>70% missing)"
                    priority = "High"
                elif col in self.numeric_cols:
                    if missing_pct < 5:
                        suggestion = "📊 Mean/Median imputation (low missing %)"
                    elif missing_pct < 20:
                        suggestion = "🧮 KNN imputation or regression-based fill"
                    else:
                        suggestion = "🔢 Forward/backward fill or create 'Missing' category"
                    priority = "Medium" if missing_pct < 20 else "Low"
                elif col in self.categorical_cols:
                    if missing_pct < 5:
                        suggestion = "🏷️ Mode imputation (most frequent value)"
                    elif missing_pct < 20:
                        suggestion = "➡️ Forward/backward fill or 'Unknown' category"
                    else:
                        suggestion = "❓ Create 'Missing' category or drop column"
                    priority = "Medium" if missing_pct < 20 else "Low"
                else:
                    suggestion = "🔍 Manual review required for this data type"
                    priority = "Manual"
                
                suggestions_data.append({
                    'Column': col,
                    'Missing %': f"{missing_pct:.1f}%",
                    'Data Type': data_type,
                    'Recommendation': suggestion,
                    'Priority': priority
                })
            
            suggestions_df = pd.DataFrame(suggestions_data)
            
            # Color-code by priority
            def highlight_priority(row):
                if row['Priority'] == 'High':
                    return ['background-color: #ffebee'] * len(row)
                elif row['Priority'] == 'Medium':
                    return ['background-color: #fff3e0'] * len(row)
                elif row['Priority'] == 'Low':
                    return ['background-color: #f3e5f5'] * len(row)
                else:
                    return ['background-color: #e8f5e8'] * len(row)
            
            st.dataframe(
                suggestions_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Implementation examples
            with st.expander("💡 Implementation Examples"):
                st.markdown("""
                **Numerical Imputation Examples:**
                ```python
                # Mean imputation
                df['column'].fillna(df['column'].mean(), inplace=True)
                
                # Median imputation (robust to outliers)
                df['column'].fillna(df['column'].median(), inplace=True)
                
                # KNN imputation
                from sklearn.impute import KNNImputer
                imputer = KNNImputer(n_neighbors=5)
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                ```
                
                **Categorical Imputation Examples:**
                ```python
                # Mode imputation
                df['column'].fillna(df['column'].mode()[0], inplace=True)
                
                # Create 'Missing' category
                df['column'].fillna('Missing', inplace=True)
                
                # Forward fill
                df['column'].fillna(method='ffill', inplace=True)
                ```
                """)
    
    def show_visual_analytics(self):
        """Comprehensive visual analytics dashboard"""
        st.markdown("## 📈 Visual Analytics Dashboard")
        
        # Debug information
        st.markdown(f"**Detected Column Types:**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.write(f"**Numerical:** {len(self.numeric_cols)}")
            if self.numeric_cols:
                st.write(", ".join(self.numeric_cols[:3]) + ("..." if len(self.numeric_cols) > 3 else ""))
        with col2:
            st.write(f"**Categorical:** {len(self.categorical_cols)}")
            if self.categorical_cols:
                st.write(", ".join(self.categorical_cols[:3]) + ("..." if len(self.categorical_cols) > 3 else ""))
        with col3:
            st.write(f"**DateTime:** {len(self.datetime_cols)}")
        with col4:
            st.write(f"**Boolean:** {len(self.boolean_cols)}")
        
        # Numerical visualizations
        if self.numeric_cols:
            st.markdown("### 🔢 Numerical Data Visualizations")
            
            # Show all numerical columns in a grid
            st.markdown("#### Distribution Analysis for All Numerical Columns")
            
            # Create visualizations for all numeric columns
            for i, col in enumerate(self.numeric_cols):
                if i % 2 == 0:
                    col1, col2 = st.columns(2)
                
                with col1 if i % 2 == 0 else col2:
                    # Histogram with box plot
                    try:
                        fig = px.histogram(
                            self.df,
                            x=col,
                            marginal="box",
                            title=f"Distribution of {col}",
                            nbins=30,
                            color_discrete_sequence=['#3498db']
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating plot for {col}: {str(e)}")
            
            # Interactive column selection for detailed analysis
            if len(self.numeric_cols) > 0:
                st.markdown("#### Detailed Analysis")
                selected_num_col = st.selectbox(
                    "Select numerical column for detailed analysis:",
                    self.numeric_cols,
                    key="num_viz_detailed"
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Box plot for outlier detection
                    fig = px.box(
                        self.df,
                        y=selected_num_col,
                        title=f"Box Plot - {selected_num_col}",
                        color_discrete_sequence=['#e74c3c']
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Violin plot
                    fig = px.violin(
                        self.df,
                        y=selected_num_col,
                        title=f"Violin Plot - {selected_num_col}",
                        color_discrete_sequence=['#9b59b6']
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Correlation analysis
            if len(self.numeric_cols) > 1:
                st.markdown("### 🔗 Correlation Analysis")
                
                # Correlation matrix
                corr_matrix = self.df[self.numeric_cols].corr()
                
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    title="Correlation Matrix",
                    color_continuous_scale='RdBu_r',
                    zmin=-1,
                    zmax=1
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Pairwise scatter plots
                if len(self.numeric_cols) >= 2:
                    st.markdown("### 📊 Pairwise Relationships")
                    
                    # Limit to 5 columns for performance
                    available_cols = self.numeric_cols[:5]
                    
                    # Select columns for scatter matrix
                    selected_cols = st.multiselect(
                        "Select columns for scatter matrix (max 5 for performance):",
                        available_cols,
                        default=available_cols[:min(3, len(available_cols))],
                        max_selections=5
                    )
                    
                    if len(selected_cols) >= 2:
                        try:
                            # Create scatter matrix
                            fig = px.scatter_matrix(
                                self.df[selected_cols].dropna(),
                                title="Scatter Matrix",
                                color_discrete_sequence=['#9b59b6']
                            )
                            fig.update_layout(height=600)
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error creating scatter matrix: {str(e)}")
        else:
            st.warning("No numerical columns detected in the dataset.")
        
        # Categorical visualizations
        if self.categorical_cols:
            st.markdown("### 🏷️ Categorical Data Visualizations")
            
            # Show visualizations for all categorical columns
            st.markdown("#### Frequency Analysis for All Categorical Columns")
            
            for i, col in enumerate(self.categorical_cols):
                if i % 2 == 0:
                    col1, col2 = st.columns(2)
                
                with col1 if i % 2 == 0 else col2:
                    try:
                        # Bar chart of value counts
                        value_counts = self.df[col].value_counts().head(10)
                        
                        if not value_counts.empty:
                            fig = px.bar(
                                x=value_counts.index,
                                y=value_counts.values,
                                title=f"Top 10 Values - {col}",
                                labels={'x': col, 'y': 'Count'},
                                color=value_counts.values,
                                color_continuous_scale='viridis'
                            )
                            fig.update_layout(height=400, xaxis_tickangle=-45)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning(f"No data found for column: {col}")
                    except Exception as e:
                        st.error(f"Error creating plot for {col}: {str(e)}")
            
            # Interactive column selection for detailed analysis
            if len(self.categorical_cols) > 0:
                st.markdown("#### Detailed Analysis")
                selected_cat_col = st.selectbox(
                    "Select categorical column for detailed analysis:",
                    self.categorical_cols,
                    key="cat_viz_detailed"
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Detailed bar chart
                    value_counts = self.df[selected_cat_col].value_counts().head(15)
                    
                    fig = px.bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        title=f"Frequency Distribution - {selected_cat_col}",
                        labels={'x': selected_cat_col, 'y': 'Count'},
                        color=value_counts.values,
                        color_continuous_scale='viridis'
                    )
                    fig.update_layout(height=400, xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Pie chart for proportions (if not too many categories)
                    if len(value_counts) <= 10:
                        fig = px.pie(
                            values=value_counts.values,
                            names=value_counts.index,
                            title=f"Proportion Distribution - {selected_cat_col}"
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Alternative visualization for high cardinality
                        st.markdown(f"**Statistics for {selected_cat_col}:**")
                        prop_df = pd.DataFrame({
                            'Category': value_counts.index,
                            'Count': value_counts.values,
                            'Percentage': (value_counts.values / len(self.df) * 100).round(2)
                        })
                        st.dataframe(prop_df, use_container_width=True, hide_index=True)
        else:
            st.warning("No categorical columns detected in the dataset.")
        
        # Combined analysis
        if self.numeric_cols and self.categorical_cols:
            st.markdown("### 🔄 Combined Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                numeric_col = st.selectbox(
                    "Select numerical column:",
                    self.numeric_cols,
                    key="combined_num"
                )
            
            with col2:
                categorical_col = st.selectbox(
                    "Select categorical column:",
                    self.categorical_cols,
                    key="combined_cat"
                )
            
            if numeric_col and categorical_col:
                try:
                    # Box plots by category
                    fig = px.box(
                        self.df,
                        x=categorical_col,
                        y=numeric_col,
                        title=f"{numeric_col} by {categorical_col}",
                        color=categorical_col
                    )
                    fig.update_layout(height=500, xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Summary statistics by group
                    group_stats = self.df.groupby(categorical_col)[numeric_col].agg([
                        'count', 'mean', 'median', 'std', 'min', 'max'
                    ]).round(3)
                    
                    st.markdown(f"**Summary Statistics of {numeric_col} by {categorical_col}:**")
                    st.dataframe(group_stats, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating combined analysis: {str(e)}")
        
        # Additional visualizations for any remaining columns
        other_cols = [col for col in self.df.columns if col not in self.numeric_cols + self.categorical_cols + self.datetime_cols + self.boolean_cols]
        if other_cols:
            st.markdown("### 🔍 Other Data Types")
            st.markdown("Columns that couldn't be automatically classified:")
            
            for col in other_cols:
                st.markdown(f"**{col}:** {self.df[col].dtype} - {self.df[col].nunique()} unique values")
                
                # Try to show some basic info
                if self.df[col].nunique() < 20:
                    st.write("Sample values:", self.df[col].value_counts().head().to_dict())
    
    def show_data_preview(self):
        """Interactive data preview with filtering options"""
        st.markdown("## 📋 Data Preview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Number of rows to display
            n_rows = st.selectbox(
                "Number of rows to display:",
                [10, 25, 50, 100, 200],
                index=0
            )
        
        with col2:
            # Data type filter
            available_types = ['All'] + ['Numerical', 'Categorical', 'DateTime', 'Boolean']
            selected_type = st.selectbox(
                "Filter by data type:",
                available_types
            )
        
        with col3:
            # Sample or head/tail
            view_type = st.selectbox(
                "View type:",
                ["Head (first rows)", "Tail (last rows)", "Random sample"]
            )
        
        # Filter columns by type
        if selected_type == 'All':
            display_cols = self.df.columns.tolist()
        elif selected_type == 'Numerical':
            display_cols = self.numeric_cols
        elif selected_type == 'Categorical':
            display_cols = self.categorical_cols
        elif selected_type == 'DateTime':
            display_cols = self.datetime_cols
        elif selected_type == 'Boolean':
            display_cols = self.boolean_cols
        
        if not display_cols:
            st.warning(f"No columns of type '{selected_type}' found in the dataset.")
            return
        
        # Additional column selection
        if len(display_cols) > 10:
            selected_cols = st.multiselect(
                "Select specific columns (leave empty for all):",
                display_cols,
                default=[]
            )
            if selected_cols:
                display_cols = selected_cols
        
        # Get data based on view type
        if view_type == "Head (first rows)":
            display_df = self.df[display_cols].head(n_rows)
        elif view_type == "Tail (last rows)":
            display_df = self.df[display_cols].tail(n_rows)
        else:  # Random sample
            display_df = self.df[display_cols].sample(min(n_rows, len(self.df)))
        
        # Display the data
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400
        )
        
        # Data summary for displayed columns
        st.markdown("### 📊 Quick Summary of Displayed Data")
        
        summary_col1, summary_col2 = st.columns(2)
        
        with summary_col1:
            st.markdown(f"""
            **Display Summary:**
            - **Rows shown:** {len(display_df):,}
            - **Columns shown:** {len(display_cols)}
            - **Total dataset size:** {len(self.df):,} × {len(self.df.columns)}
            """)
        
        with summary_col2:
            if display_cols:
                numeric_display_cols = [col for col in display_cols if col in self.numeric_cols]
                if numeric_display_cols:
                    st.markdown("**Numerical Columns Summary:**")
                    quick_stats = display_df[numeric_display_cols].describe().round(2)
                    st.dataframe(quick_stats, use_container_width=True)
    
    def get_summary_statistics(self):
        """Generate comprehensive summary statistics for export"""
        summary_data = {}
        
        # Basic info
        summary_data['Dataset Info'] = pd.Series({
            'Total Rows': len(self.df),
            'Total Columns': len(self.df.columns),
            'Memory Usage (MB)': self.df.memory_usage(deep=True).sum() / (1024 * 1024),
            'Missing Values': self.df.isnull().sum().sum(),
            'Missing Percentage': (self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100,
            'Duplicate Rows': self.df.duplicated().sum()
        })
        
        # Numerical statistics
        if self.numeric_cols:
            numeric_stats = self.df[self.numeric_cols].describe()
            # Add skewness and kurtosis
            numeric_stats.loc['skewness'] = self.df[self.numeric_cols].skew()
            numeric_stats.loc['kurtosis'] = self.df[self.numeric_cols].kurtosis()
            summary_data['Numerical Statistics'] = numeric_stats
        
        # Categorical statistics
        if self.categorical_cols:
            cat_stats_list = []
            for col in self.categorical_cols:
                col_stats = {
                    'Column': col,
                    'Count': self.df[col].count(),
                    'Unique': self.df[col].nunique(),
                    'Top': self.df[col].mode().iloc[0] if not self.df[col].mode().empty else 'N/A',
                    'Freq': self.df[col].value_counts().iloc[0] if len(self.df[col].value_counts()) > 0 else 0
                }
                cat_stats_list.append(col_stats)
            summary_data['Categorical Statistics'] = pd.DataFrame(cat_stats_list)
        
        # Missing values analysis
        missing_stats = pd.DataFrame({
            'Column': self.df.columns,
            'Missing Count': self.df.isnull().sum(),
            'Missing Percentage': (self.df.isnull().sum() / len(self.df)) * 100,
            'Data Type': self.df.dtypes.astype(str)
        })
        summary_data['Missing Values Analysis'] = missing_stats
        
        # Combine all summaries
        combined_summary = pd.concat([
            pd.DataFrame({'Metric': summary_data['Dataset Info'].index, 'Value': summary_data['Dataset Info'].values})
        ], ignore_index=True)
        
        return combined_summary
