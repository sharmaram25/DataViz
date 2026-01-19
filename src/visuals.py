import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Define a professional color palette using the "Slate" theme inspiration
# Using a custom sequence that complements the #2563EB primary color
COLOR_PALETTE = ["#2563EB", "#3B82F6", "#60A5FA", "#93C5FD", "#1E40AF", "#1E3A8A"]
TEMPLATE = "plotly_white"
MAX_POINTS = 5000  # Downsample threshold for 3G optimization

def sample_data(df, n=MAX_POINTS):
    """
    Downsamples the dataframe for visualization if rows > n.
    Returns sampled dataframe.
    """
    if len(df) > n:
        return df.sample(n, random_state=42)
    return df

def apply_chart_style(fig, title="", height=None):
    """
    Applies consistent premium styling to Plotly figures.
    """
    fig.update_layout(
        template=TEMPLATE,
        title={
            'text': f"<b>{title}</b>",
            'y':0.96,
            'x':0,
            'xanchor': 'left',
            'yanchor': 'top',
            'font': dict(size=18, color="#0F172A", family="Inter, sans-serif")
        },
        font=dict(family="Inter, sans-serif", color="#64748B"),
        margin=dict(t=60, l=20, r=20, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)", 
        height=height,
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Inter, sans-serif"
        )
    )
    # Refined gridlines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#F1F5F9', zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#F1F5F9', zeroline=False)
    return fig

def plot_histogram(df, x, color=None, bins=30):
    fig = px.histogram(
        df, x=x, color=color, nbins=bins, 
        color_discrete_sequence=COLOR_PALETTE,
        opacity=0.8,
        marginal="box" # Adds a boxplot on top
    )
    return apply_chart_style(fig, title=f"Distribution Analysis: {x}")

def plot_scatter(df, x, y, color=None, size=None, trendline=False, animation_frame=None, facet_col=None, log_x=False, log_y=False):
    df_plot = sample_data(df)
    trend = "ols" if trendline else None
    
    # Handle dates if applicable
    if pd.api.types.is_datetime64_any_dtype(df_plot[x]):
        df_plot = df_plot.sort_values(x)
        
    fig = px.scatter(
        df_plot, x=x, y=y, color=color, size=size,
        color_discrete_sequence=COLOR_PALETTE,
        trendline=trend,
        animation_frame=animation_frame,
        facet_col=facet_col,
        log_x=log_x, log_y=log_y,
        hover_data=df_plot.columns.tolist()[:5]
    )
    fig.update_traces(marker=dict(line=dict(width=0.5, color='white')))
    title_suffix = " (Sampled)" if len(df) > MAX_POINTS else ""
    return apply_chart_style(fig, title=f"Correlation: {y} vs {x}{title_suffix}")

def plot_density_heatmap(df, x, y, facet_col=None, animation_frame=None):
    df_plot = sample_data(df)
    fig = px.density_heatmap(
        df_plot, x=x, y=y, 
        facet_col=facet_col, 
        animation_frame=animation_frame,
        color_continuous_scale="Viridis"
    )
    return apply_chart_style(fig, title=f"Density Heatmap: {x} vs {y}")

def plot_density_contour(df, x, y, color=None):
    df_plot = sample_data(df)
    fig = px.density_contour(
        df_plot, x=x, y=y, color=color,
        color_discrete_sequence=COLOR_PALETTE
    )
    fig.update_traces(contours_coloring="fill", contours_showlabels=True)
    return apply_chart_style(fig, title=f"Density Contours: {x} vs {y}")

def plot_sunburst(df, path, values=None, color=None):
    # Sunburst requires no sampling usually as it aggregates, but let's be safe if huge
    # Actually sunburst aggregates automatically, so we pass full data usually.
    # But if path has high cardinality, it gets messy. 
    fig = px.sunburst(
        df, path=path, values=values, color=color,
        color_discrete_sequence=COLOR_PALETTE
    )
    return apply_chart_style(fig, title=f"Hierarchical View: {' -> '.join(path)}")

def plot_treemap(df, path, values=None, color=None):
    fig = px.treemap(
        df, path=path, values=values, color=color,
        color_discrete_sequence=COLOR_PALETTE
    )
    return apply_chart_style(fig, title=f"Treemap Breakdown: {' -> '.join(path)}")

def plot_funnel(df, x, y, color=None):
    fig = px.funnel(
        df, x=x, y=y, color=color,
        color_discrete_sequence=COLOR_PALETTE
    )
    return apply_chart_style(fig, title=f"Funnel Analysis: {y} by {x}")

def plot_3d_scatter(df, x, y, z, color=None):
    df_plot = sample_data(df)
    fig = px.scatter_3d(
        df_plot, x=x, y=y, z=z, color=color,
        color_discrete_sequence=COLOR_PALETTE
    )
    title_suffix = " (Sampled)" if len(df) > MAX_POINTS else ""
    return apply_chart_style(fig, title=f"3D Analysis: {x}, {y}, {z}{title_suffix}")

def plot_violin(df, x, y, color=None):
    df_plot = sample_data(df)
    fig = px.violin(
        df_plot, x=x, y=y, color=color, box=True, points="all",
        color_discrete_sequence=COLOR_PALETTE
    )
    title_suffix = " (Sampled)" if len(df) > MAX_POINTS else ""
    return apply_chart_style(fig, title=f"Violin Distribution: {y} by {x}{title_suffix}")

def plot_box(df, x, y, color=None):
    df_plot = sample_data(df)
    fig = px.box(
        df_plot, x=x, y=y, color=color,
        color_discrete_sequence=COLOR_PALETTE,
    )
    title_suffix = " (Sampled)" if len(df) > MAX_POINTS else ""
    return apply_chart_style(fig, title=f"Box Plot: {y} by {x}{title_suffix}")

def plot_heatmap(df, matrix, title="Correlation Heatmap"):
    fig = px.imshow(
        matrix, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu_r",
        origin='lower',
        zmin=-1, zmax=1
    )
    return apply_chart_style(fig, title=title)

def plot_missing_matrix(df):
    """
    Visualizes missing values pattern.
    """
    # Create a boolean dataframe (True if missing)
    missing_df = df.isnull()
    
    # Downsample if needed (cols are important, rows can be sampled)
    if len(missing_df) > 2000:
        missing_df = missing_df.sample(2000, random_state=42)
        
    fig = px.imshow(
        missing_df, 
        color_continuous_scale=["#E2E8F0", "#EF4444"], # Slate-200 to Red-500
        aspect="auto",
        labels=dict(x="Column", y="Row Index", color="Is Missing")
    )
    # Customize axis to look cleaner
    fig.update_layout(yaxis={'visible': False, 'showticklabels': False})
    return apply_chart_style(fig, title="Missing Data Pattern (Red = Missing)")

def plot_bar(df, x, y, color=None):
    title = f"{y} by {x}" if y else f"Count of {x}"
    fig = px.bar(
        df, x=x, y=y, color=color,
        color_discrete_sequence=COLOR_PALETTE,
    )
    return apply_chart_style(fig, title=title)

def plot_line(df, x, y, color=None):
    fig = px.line(
        df, x=x, y=y, color=color,
        color_discrete_sequence=COLOR_PALETTE,
        markers=True
    )
    fig.update_layout(hovermode="x unified")
    return apply_chart_style(fig, title=f"Trend: {y} over {x}")
