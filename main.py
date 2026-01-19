import streamlit as st
import pandas as pd
import plotly.express as px
from src.loader import load_data
from src.preprocess import DataPreprocessor
from src.eda import EDAAnalyzer
from src.visuals import (
    plot_histogram, plot_scatter, plot_box, plot_heatmap, 
    plot_bar, plot_line, plot_3d_scatter, plot_violin,
    plot_missing_matrix, apply_chart_style,
    plot_density_heatmap, plot_density_contour, plot_sunburst, plot_treemap, plot_funnel
)
from src.ui_components import render_header, render_sidebar, render_metrics, setup_page
from src.utils import convert_df_to_csv, generate_eda_report

# --- Page Setup ---
st.set_page_config(layout="wide", page_title="DataViz", page_icon="📊")
setup_page()

# --- Session State ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'file_name' not in st.session_state:
    st.session_state.file_name = None

# --- Main App Logic ---

def main():
    
    if st.session_state.df is None:
        # LANDING AREA
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            render_header()
            
            uploaded_file = st.file_uploader(
                "📂 Drag and drop your dataset to begin analysis", 
                type=['csv', 'xlsx', 'json']
            )
            
            if uploaded_file:
                with st.spinner("🚀 Ingesting & Analyzing data..."):
                    df, error = load_data(uploaded_file)
                    
                    if error:
                        st.error(error)
                    else:
                        st.session_state.df = df
                        st.session_state.file_name = uploaded_file.name
                        st.rerun()
                        
    else:
        # APP AREA
        page = render_sidebar()
        
        # Sidebar Controls
        if st.sidebar.button("🗑️ Unload Dataset", type="primary"):
            st.session_state.df = None
            st.session_state.file_name = None
            st.rerun()
            
        st.sidebar.markdown("---")
        st.sidebar.subheader("📥 Export Results")
        
        # Initialize Core Objects Shared Across Pages
        df = st.session_state.df
        preprocessor = DataPreprocessor(df)
        eda = EDAAnalyzer(df)
        
        # Export Controls
        st.sidebar.download_button(
            "💾 Download CSV",
            data=convert_df_to_csv(df),
            file_name=f"{st.session_state.file_name}_processed.csv",
            mime="text/csv"
        )
        
        report_md = generate_eda_report(df, eda, preprocessor)
        st.sidebar.download_button(
            "📄 Download EDA Report",
            data=report_md,
            file_name=f"{st.session_state.file_name}_report.md",
            mime="text/markdown"
        )
        
        
        # --- PAGE: OVERVIEW ---
        if page == "Dataset Overview":
            st.title(f"Overview: {st.session_state.file_name}")
            
            # --- FILTERING SECTION ---
            with st.expander("🔎 Filter Data", expanded=False):
                st.info("Filtering affects Export but not current global Metrics (for safety).")
            
            # Metrics
            summary = preprocessor.get_summary()
            score = preprocessor.get_data_quality_score()
            render_metrics(summary, score)
            
            st.markdown("### 📋 Data Preview")
            
            preview_tabs = st.tabs(["Header (First 50)", "Tail (Last 50)", "Structure & Types"])
            
            with preview_tabs[0]:
                st.dataframe(df.head(50), use_container_width=True, height=400)
            
            with preview_tabs[1]:
                st.dataframe(df.tail(50), use_container_width=True, height=400)
                
            with preview_tabs[2]:
                # Custom Structure Dataframe
                dtypes = df.dtypes.astype(str).reset_index()
                dtypes.columns = ["Column", "Type"]
                dtypes["Missing %"] = df.isnull().mean().values * 100
                dtypes["Unique Values"] = df.nunique().values
                st.dataframe(dtypes, use_container_width=True)

            # Detailed Statistics Section
            st.markdown("### 📊 Key Statistics")
            stats_tabs = st.tabs(["Numerical Summary", "Categorical Summary"])
            
            with stats_tabs[0]:
                nums = df.select_dtypes(include=[float, int])
                if not nums.empty:
                    st.dataframe(nums.describe().T.style.background_gradient(cmap="Blues"), use_container_width=True)
                else:
                    st.info("No numerical columns found.")
            
            with stats_tabs[1]:
                cats = df.select_dtypes(include=[object, "category"])
                if not cats.empty:
                    st.dataframe(cats.describe().T, use_container_width=True)
                else:
                    st.info("No categorical columns found.")

            # Quick Actions Section (Cleaned up)
            st.markdown("---")
            c1, c2 = st.columns([3, 1])
            with c1:
                st.info("💡 **Pro Tip**: Use 'Advanced Exploration' for deep insights, PCA, and Clustering.")
            with c2:
                 st.download_button(
                    "💾 Download Processed CSV",
                    data=convert_df_to_csv(df),
                    file_name=f"{st.session_state.file_name}_processed.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        # --- PAGE: ADVANCED EXPLORATION ---
        # --- PAGE: ADVANCED EXPLORATION ---
        elif page == "Advanced Exploration":
            st.title("🔎 Deep Dive Analysis")
            
            # Identify columns types for easy selection
            nums = preprocessor.get_column_types()['numerical']
            cats = preprocessor.get_column_types()['categorical']
            dates = preprocessor.get_column_types()['datetime']
            
            adv_tabs = st.tabs([
                "🏥 Data Quality", 
                "📊 Univariate & Normality", 
                "🔗 Bivariate & Target Analysis", 
                "📈 Time Series", 
                "🧬 Multivariate & PCA"
            ])
            
            # 1. Data Quality
            with adv_tabs[0]:
                st.markdown("#### Missing Values Pattern")
                st.plotly_chart(plot_missing_matrix(df), use_container_width=True)
                
                st.markdown("#### Smart Diagnostics")
                insights = eda.get_smart_insights()
                if insights:
                    for insight in insights:
                        st.warning(insight) if "⚠️" in insight else st.info(insight)
                else:
                    st.success("✅ No critical data quality issues found.")
            
            # 2. Univariate
            with adv_tabs[1]:
                st.markdown("#### Distribution Analysis")
                if nums:
                    target_col = st.selectbox("Select Numerical Variable", options=nums, key="uni_select")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.plotly_chart(plot_histogram(df, target_col), use_container_width=True)
                    with c2:
                        st.plotly_chart(plot_violin(df, x=None, y=target_col), use_container_width=True)
                        
                    # Normality Test
                    st.markdown("##### 🧪 Normality Test (D'Agostino's K^2)")
                    k2, p, is_normal = eda.test_normality(target_col)
                    if k2 is not None:
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Statistic", f"{k2:.2f}")
                        col2.metric("P-Value", f"{p:.4f}")
                        col3.metric("Is Normal (p>0.05)?", "✅ Yes" if is_normal else "❌ No")
                        
                        if not is_normal:
                            st.caption("Distribution deviates from Gaussian. Consider Log-Transform or Non-Parametric tests.")
                    else:
                        st.info("Not enough data for normality test.")
                else:
                    st.info("No numerical data available.")
                    
            # 3. Bivariate / Target
            with adv_tabs[2]:
                st.markdown("#### Target Correlation")
                if nums:
                    target = st.selectbox("Select Target Variable to Analyze", options=nums, key="target_select")
                    corr_df = eda.compute_target_correlation(target)
                    
                    if not corr_df.empty:
                        # Visualize correlations
                        fig_corr = px.bar(
                            corr_df.drop(0), # Drop self-correlation
                            x="Correlation", y="Feature", 
                            color="Correlation",
                            color_continuous_scale="RdBu",
                            title=f"Feature Correlation with {target}",
                            orientation='h'
                        )
                        st.plotly_chart(apply_chart_style(fig_corr), use_container_width=True)
                    else:
                        st.warning("Could not compute correlations.")
                else:
                    st.info("Need numerical columns for correlation analysis.")
            
            # 4. Time Series
            with adv_tabs[3]:
                if dates:
                    date_col = st.selectbox("Select Date Column", options=dates)
                    val_col = st.selectbox("Select Value Column", options=nums) if nums else None
                    
                    if date_col and val_col:
                        # Simple resampling trend
                        try:
                            ts_df = df.set_index(pd.to_datetime(df[date_col])).sort_index()
                            # Auto-sample or aggregate
                            ts_resampled = ts_df[val_col].resample('M').mean().reset_index()
                            
                            st.markdown(f"#### Monthly Trend: {val_col}")
                            fig_ts = px.line(ts_resampled, x=date_col, y=val_col, markers=True)
                            st.plotly_chart(apply_chart_style(fig_ts), use_container_width=True)
                        except Exception as e:
                            st.error(f"Could not process time series: {e}")
                else:
                    st.info("No Datetime columns detected in this dataset.")

            # 5. Multivariate
            with adv_tabs[4]:
                st.markdown("#### Correlation Matrix")
                corr = eda.compute_correlation_matrix()
                if corr is not None:
                    st.plotly_chart(plot_heatmap(df, corr), use_container_width=True)
                
                st.markdown("#### PCA (3D Structure)")
                pca_df, explained_var = eda.run_pca_analysis()
                if pca_df is not None:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.plotly_chart(
                            plot_3d_scatter(pca_df, 'PC1', 'PC2', 'PC3', color='Color_Dimension'),
                            use_container_width=True
                        )
                    with col2:
                        st.metric("Var (PC1)", f"{explained_var[0]:.2%}")
                        st.metric("Var (PC2)", f"{explained_var[1]:.2%}")
                        st.metric("Var (PC3)", f"{explained_var[2]:.2%}")

        # --- PAGE: VISUALIZATION STUDIO 2.0 ---
        elif page == "Visualization Studio":
            st.title("🎨 Visualization Studio 2.0")
            
            # Layout: Left Control Panel (25%), Right Canvas (75%)
            row1 = st.container()
            with row1:
                c_control, c_canvas = st.columns([1, 3])
                
            with c_control:
                st.subheader("⚙️ Studio Config")
                
                # 1. Chart Family Selection
                chart_family = st.selectbox(
                    "Chart Family",
                    ["Correlation & Relationship", "Distribution & Density", "Hierarchy & Composition", "Comparison & Trend"],
                    index=0
                )
                
                # 2. Specific Chart Type based on Family
                chart_type = None
                if chart_family == "Correlation & Relationship":
                    chart_type = st.radio("Chart Type", ["Scatter Plot", "3D Scatter", "Bubble Chart"], horizontal=True)
                elif chart_family == "Distribution & Density":
                    chart_type = st.radio("Chart Type", ["Histogram", "Violin Plot", "Box Plot", "Density Heatmap", "Density Contour"], horizontal=True)
                elif chart_family == "Hierarchy & Composition":
                    chart_type = st.radio("Chart Type", ["Sunburst", "Treemap", "Pie Chart"], horizontal=True)
                elif chart_family == "Comparison & Trend":
                    chart_type = st.radio("Chart Type", ["Bar Chart", "Line Chart", "Funnel"], horizontal=True)
                    
                st.markdown("---")
                
                # 3. Data Selectors (Dynamic based on chart type)
                cols = df.columns.tolist()
                numerics = preprocessor.get_column_types()['numerical']
                cats = preprocessor.get_column_types()['categorical']
                
                x_axis, y_axis, z_axis = None, None, None
                color_col, size_col, facet_col = None, None, None
                anim_col = None
                path_cols = []
                
                # --- INPUT LOGIC ---
                if chart_family in ["Correlation & Relationship", "Comparison & Trend", "Distribution & Density"]:
                    if chart_type not in ["Sunburst", "Treemap", "Funnel"]:
                        x_axis = st.selectbox("X-Axis", options=cols, index=0)
                        
                    if chart_type not in ["Histogram", "Density Contour", "Pie Chart"]:
                        y_axis = st.selectbox("Y-Axis", options=cols, index=1 if len(cols)>1 else 0)
                        
                    color_col = st.selectbox("Color / Group", options=[None] + cols, help="Map color to a variable")
                    
                    if chart_type in ["Scatter Plot", "Bubble Chart", "3D Scatter"]:
                        size_col = st.selectbox("Size Attribute", options=[None] + numerics, help="Map bubble size to a variable")
                        
                    if chart_type == "3D Scatter":
                        z_axis = st.selectbox("Z-Axis", options=cols, index=2 if len(cols)>2 else 0)
                        
                    # Advanced Scatter Controls
                    if chart_type == "Scatter Plot" or chart_type == "Bubble Chart":
                        with st.expander("🚀 Advanced Scatter Options"):
                            facet_col = st.selectbox("Facet Column (Grid)", options=[None] + cats)
                            anim_col = st.selectbox("Animation Frame (Time/Seq)", options=[None] + cols)
                            log_x = st.checkbox("Log Scale X", value=False)
                            log_y = st.checkbox("Log Scale Y", value=False)
                            trend = st.checkbox("Show Trendline (OLS)", value=False)
                            
                    if chart_type == "Density Heatmap":
                         with st.expander("🔥 Heatmap Options"):
                             facet_col = st.selectbox("Facet Column", options=[None] + cats)
                             anim_col = st.selectbox("Animation Frame", options=[None] + cols)

                elif chart_family == "Hierarchy & Composition":
                    if chart_type in ["Sunburst", "Treemap"]:
                        st.info("Select hierarchy levels (e.g. Region -> State -> City)")
                        path_cols = st.multiselect("Hierarchy Path (Order matters!)", options=cats, default=cats[:2] if len(cats)>=2 else cats[:1])
                        values_col = st.selectbox("Size Value", options=[None] + numerics)
                        color_col = st.selectbox("Color Value", options=[None] + numerics + cats)
                        
            # --- RENDER LOGIC (Right Canvas) ---
            with c_canvas:
                try:
                    fig = None
                    
                    # SCATTER FAMILY
                    if chart_type in ["Scatter Plot", "Bubble Chart"]:
                        if x_axis and y_axis:
                            fig = plot_scatter(
                                df, x_axis, y_axis, color=color_col, size=size_col, 
                                trendline=trend if 'trend' in locals() else False,
                                animation_frame=anim_col if 'anim_col' in locals() else None,
                                facet_col=facet_col if 'facet_col' in locals() else None,
                                log_x=log_x if 'log_x' in locals() else False,
                                log_y=log_y if 'log_y' in locals() else False
                            )
                    elif chart_type == "3D Scatter":
                        if x_axis and y_axis and z_axis:
                            fig = plot_3d_scatter(df, x_axis, y_axis, z_axis, color=color_col)
                            
                    # DISTRIBUTION FAMILY
                    elif chart_type == "Histogram":
                        if x_axis:
                            fig = plot_histogram(df, x_axis, color=color_col)
                    elif chart_type == "Violin Plot":
                        if x_axis: # can be just y or x and y
                            fig = plot_violin(df, x=x_axis, y=y_axis, color=color_col)
                    elif chart_type == "Box Plot":
                         if x_axis:
                             fig = plot_box(df, x_axis, y_axis, color=color_col)
                    elif chart_type == "Density Heatmap":
                        if x_axis and y_axis:
                            fig = plot_density_heatmap(
                                df, x_axis, y_axis, 
                                facet_col=facet_col if 'facet_col' in locals() else None,
                                animation_frame=anim_col if 'anim_col' in locals() else None
                            )
                    elif chart_type == "Density Contour":
                         if x_axis and y_axis:
                             fig = plot_density_contour(df, x_axis, y_axis, color=color_col)

                    # HIERARCHY FAMILY
                    elif chart_type == "Sunburst":
                        if path_cols:
                            fig = plot_sunburst(df, path=path_cols, values=values_col, color=color_col)
                        else:
                            st.warning("Please select at least 1 hierarchy level.")
                    elif chart_type == "Treemap":
                         if path_cols:
                            fig = plot_treemap(df, path=path_cols, values=values_col, color=color_col)
                    
                    # COMPARISON FAMILY
                    elif chart_type == "Bar Chart":
                        fig = plot_bar(df, x_axis, y_axis, color=color_col)
                    elif chart_type == "Line Chart":
                        fig = plot_line(df, x_axis, y_axis, color=color_col)
                    elif chart_type == "Funnel":
                         if x_axis and y_axis:
                             # funnel usually x=values, y=stages
                             fig = plot_funnel(df, x_axis, y_axis, color=color_col)
                             
                    # RENDER
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    elif not fig and chart_type:
                        st.info("👈 Configure chart options to generate visualization.")
                        
                except Exception as e:
                    st.error(f"Could not render {chart_type}: {e}")

if __name__ == "__main__":
    main()
