"""
Report Generator - HTML and PDF export functionality
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from datetime import datetime
import base64
import io
from pathlib import Path

class ReportGenerator:
    """Generate comprehensive EDA reports in HTML and PDF formats"""
    
    def __init__(self, df, filename):
        self.df = df
        self.filename = filename
        self.generated_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Column types
        self.numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        self.boolean_cols = df.select_dtypes(include=['bool']).columns.tolist()
    
    def generate_html_report(self, analyzer):
        """Generate comprehensive HTML report"""
        
        # Generate plots
        plots_html = self._generate_plots_html()
        
        # Generate statistics tables
        stats_html = self._generate_statistics_html()
        
        # Generate missing values analysis
        missing_html = self._generate_missing_values_html()
        
        # Main HTML template
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>DataViz EDA Report - {self.filename}</title>
            <style>
                {self._get_css_styles()}
            </style>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <div class="container">
                {self._generate_header()}
                {self._generate_executive_summary()}
                {stats_html}
                {missing_html}
                {plots_html}
                {self._generate_recommendations()}
                {self._generate_footer()}
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    def generate_pdf_report(self, analyzer):
        """Generate PDF report using xhtml2pdf"""
        try:
            from xhtml2pdf import pisa
            
            # Generate simplified HTML for PDF
            html_content = self._generate_pdf_html()
            
            # Convert HTML to PDF
            result = io.BytesIO()
            pdf = pisa.pisaDocument(io.BytesIO(html_content.encode("UTF-8")), result)
            
            if not pdf.err:
                return result.getvalue()
            else:
                raise Exception("PDF generation failed")
                
        except ImportError:
            # Fallback to simple PDF generation
            return self._generate_simple_pdf(analyzer)
    
    def _generate_header(self):
        """Generate HTML header section"""
        return f"""
        <header class="header">
            <div class="header-content">
                <h1>🧠 DataViz - Exploratory Data Analysis Report</h1>
                <div class="report-info">
                    <p><strong>Dataset:</strong> {self.filename}</p>
                    <p><strong>Generated:</strong> {self.generated_time}</p>
                    <p><strong>Rows:</strong> {len(self.df):,} | <strong>Columns:</strong> {len(self.df.columns)}</p>
                </div>
            </div>
        </header>
        """
    
    def _generate_executive_summary(self):
        """Generate executive summary section"""
        total_missing = self.df.isnull().sum().sum()
        missing_percentage = (total_missing / (len(self.df) * len(self.df.columns))) * 100
        duplicate_count = self.df.duplicated().sum()
        
        return f"""
        <section class="executive-summary">
            <h2>📋 Executive Summary</h2>
            <div class="summary-grid">
                <div class="summary-card">
                    <h3>Dataset Overview</h3>
                    <ul>
                        <li><strong>Shape:</strong> {len(self.df):,} rows × {len(self.df.columns)} columns</li>
                        <li><strong>Memory Usage:</strong> {self.df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB</li>
                        <li><strong>Numerical Columns:</strong> {len(self.numeric_cols)}</li>
                        <li><strong>Categorical Columns:</strong> {len(self.categorical_cols)}</li>
                    </ul>
                </div>
                <div class="summary-card">
                    <h3>Data Quality</h3>
                    <ul>
                        <li><strong>Missing Values:</strong> {total_missing:,} ({missing_percentage:.1f}%)</li>
                        <li><strong>Duplicate Rows:</strong> {duplicate_count:,}</li>
                        <li><strong>Complete Rows:</strong> {len(self.df.dropna()):,}</li>
                    </ul>
                </div>
            </div>
        </section>
        """
    
    def _generate_statistics_html(self):
        """Generate statistics section"""
        html = "<section class='statistics-section'><h2>📊 Statistical Analysis</h2>"
        
        # Numerical statistics
        if self.numeric_cols:
            numeric_stats = self.df[self.numeric_cols].describe()
            html += f"""
            <div class="stats-container">
                <h3>Numerical Features Statistics</h3>
                {numeric_stats.to_html(classes='stats-table')}
            </div>
            """
        
        # Categorical statistics
        if self.categorical_cols:
            cat_stats_list = []
            for col in self.categorical_cols[:10]:  # Limit to first 10 for report
                stats = {
                    'Column': col,
                    'Count': self.df[col].count(),
                    'Unique': self.df[col].nunique(),
                    'Top Value': self.df[col].mode().iloc[0] if not self.df[col].mode().empty else 'N/A',
                    'Frequency': self.df[col].value_counts().iloc[0] if len(self.df[col].value_counts()) > 0 else 0
                }
                cat_stats_list.append(stats)
            
            cat_stats_df = pd.DataFrame(cat_stats_list)
            html += f"""
            <div class="stats-container">
                <h3>Categorical Features Statistics</h3>
                {cat_stats_df.to_html(classes='stats-table', index=False)}
            </div>
            """
        
        html += "</section>"
        return html
    
    def _generate_missing_values_html(self):
        """Generate missing values analysis section"""
        missing_counts = self.df.isnull().sum()
        missing_percentages = (missing_counts / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing_counts.index,
            'Missing Count': missing_counts.values,
            'Missing %': missing_percentages.round(2)
        })
        
        # Filter to show only columns with missing values
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False)
        
        if missing_df.empty:
            return """
            <section class='missing-section'>
                <h2>❓ Missing Values Analysis</h2>
                <div class="success-message">
                    <h3>🎉 No Missing Values Found!</h3>
                    <p>Your dataset is complete and ready for analysis.</p>
                </div>
            </section>
            """
        
        return f"""
        <section class='missing-section'>
            <h2>❓ Missing Values Analysis</h2>
            <div class="missing-container">
                <h3>Missing Values Summary</h3>
                {missing_df.to_html(classes='stats-table', index=False)}
            </div>
        </section>
        """
    
    def _generate_plots_html(self):
        """Generate plots section"""
        plots_html = "<section class='plots-section'><h2>📈 Visualizations</h2>"
        
        # Numerical distribution plots
        if self.numeric_cols:
            plots_html += "<h3>Numerical Distributions</h3><div class='plots-grid'>"
            
            for col in self.numeric_cols[:4]:  # Limit to first 4 columns
                fig = px.histogram(
                    self.df,
                    x=col,
                    title=f"Distribution of {col}",
                    marginal="box"
                )
                fig.update_layout(height=400, showlegend=False)
                plots_html += f"<div class='plot-container'>{pio.to_html(fig, include_plotlyjs=False, div_id=f'plot_{col}')}</div>"
            
            plots_html += "</div>"
        
        # Categorical distribution plots
        if self.categorical_cols:
            plots_html += "<h3>Categorical Distributions</h3><div class='plots-grid'>"
            
            for col in self.categorical_cols[:4]:  # Limit to first 4 columns
                value_counts = self.df[col].value_counts().head(10)
                fig = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    title=f"Distribution of {col}",
                    labels={'x': col, 'y': 'Count'}
                )
                fig.update_layout(height=400, showlegend=False, xaxis_tickangle=-45)
                plots_html += f"<div class='plot-container'>{pio.to_html(fig, include_plotlyjs=False, div_id=f'plot_cat_{col}')}</div>"
            
            plots_html += "</div>"
        
        # Correlation heatmap for numerical columns
        if len(self.numeric_cols) > 1:
            corr_matrix = self.df[self.numeric_cols].corr()
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                title="Correlation Matrix",
                color_continuous_scale='RdBu_r'
            )
            fig.update_layout(height=500)
            plots_html += f"<h3>Correlation Analysis</h3><div class='plot-container full-width'>{pio.to_html(fig, include_plotlyjs=False, div_id='correlation_plot')}</div>"
        
        plots_html += "</section>"
        return plots_html
    
    def _generate_recommendations(self):
        """Generate recommendations section"""
        recommendations = []
        
        # Data quality recommendations
        total_missing = self.df.isnull().sum().sum()
        if total_missing > 0:
            recommendations.append("📝 Address missing values using appropriate imputation strategies")
        
        duplicate_count = self.df.duplicated().sum()
        if duplicate_count > 0:
            recommendations.append(f"🔄 Remove {duplicate_count:,} duplicate rows")
        
        # Column-specific recommendations
        constant_cols = [col for col in self.df.columns if self.df[col].nunique() <= 1]
        if constant_cols:
            recommendations.append(f"🗑️ Consider removing {len(constant_cols)} constant columns")
        
        # High correlation recommendations
        if len(self.numeric_cols) > 1:
            corr_matrix = self.df[self.numeric_cols].corr()
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.95:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
            
            if high_corr_pairs:
                recommendations.append(f"🔗 Review {len(high_corr_pairs)} highly correlated column pairs")
        
        # Data type recommendations
        large_categorical = [col for col in self.categorical_cols if self.df[col].nunique() > 50]
        if large_categorical:
            recommendations.append("🏷️ Consider encoding or grouping high-cardinality categorical variables")
        
        if not recommendations:
            recommendations.append("✅ Your dataset appears to be in good shape for analysis!")
        
        recommendations_html = """
        <section class='recommendations-section'>
            <h2>💡 Recommendations</h2>
            <div class='recommendations-list'>
        """
        
        for rec in recommendations:
            recommendations_html += f"<div class='recommendation-item'>{rec}</div>"
        
        recommendations_html += "</div></section>"
        return recommendations_html
    
    def _generate_footer(self):
        """Generate footer section"""
        return f"""
        <footer class='footer'>
            <div class='footer-content'>
                <p>Generated by DataViz - Intelligent EDA Platform</p>
                <p>Report created on {self.generated_time}</p>
                <p>Dataset: {self.filename} | Rows: {len(self.df):,} | Columns: {len(self.df.columns)}</p>
            </div>
        </footer>
        """
    
    def _get_css_styles(self):
        """Get CSS styles for HTML report"""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #ecf0f1, #bdc3c7);
            color: #2c3e50;
            line-height: 1.6;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
            border-radius: 10px;
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #2c3e50, #34495e);
            color: white;
            padding: 2rem;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 1rem;
        }
        
        .report-info {
            display: flex;
            justify-content: center;
            gap: 2rem;
            flex-wrap: wrap;
        }
        
        .report-info p {
            background: rgba(255,255,255,0.1);
            padding: 0.5rem 1rem;
            border-radius: 5px;
        }
        
        section {
            padding: 2rem;
            border-bottom: 1px solid #ecf0f1;
        }
        
        h2 {
            color: #2c3e50;
            font-size: 2rem;
            margin-bottom: 1.5rem;
            border-bottom: 3px solid #3498db;
            padding-bottom: 0.5rem;
        }
        
        h3 {
            color: #34495e;
            font-size: 1.5rem;
            margin-bottom: 1rem;
        }
        
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 2rem;
            margin-top: 1rem;
        }
        
        .summary-card {
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 4px solid #3498db;
        }
        
        .summary-card h3 {
            color: #2c3e50;
            margin-bottom: 1rem;
        }
        
        .summary-card ul {
            list-style: none;
            padding-left: 0;
        }
        
        .summary-card li {
            padding: 0.25rem 0;
            border-bottom: 1px solid #dee2e6;
        }
        
        .stats-table {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .stats-table th {
            background: linear-gradient(135deg, #34495e, #2c3e50);
            color: white;
            padding: 1rem;
            text-align: left;
            font-weight: bold;
        }
        
        .stats-table td {
            padding: 0.75rem 1rem;
            border-bottom: 1px solid #ecf0f1;
        }
        
        .stats-table tr:nth-child(even) {
            background: #f8f9fa;
        }
        
        .stats-table tr:hover {
            background: #e3f2fd;
        }
        
        .plots-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 2rem;
            margin: 2rem 0;
        }
        
        .plot-container {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 1rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .plot-container.full-width {
            grid-column: 1 / -1;
        }
        
        .success-message {
            background: linear-gradient(135deg, #d4edda, #c3e6cb);
            border: 1px solid #c3e6cb;
            border-radius: 10px;
            padding: 2rem;
            text-align: center;
            color: #155724;
        }
        
        .success-message h3 {
            color: #155724;
            font-size: 2rem;
            margin-bottom: 1rem;
        }
        
        .recommendations-list {
            display: grid;
            gap: 1rem;
        }
        
        .recommendation-item {
            background: linear-gradient(135deg, #fff3cd, #ffeaa7);
            border-left: 4px solid #f39c12;
            padding: 1rem;
            border-radius: 5px;
            font-weight: 500;
        }
        
        .footer {
            background: linear-gradient(135deg, #34495e, #2c3e50);
            color: white;
            padding: 2rem;
            text-align: center;
        }
        
        .footer-content p {
            margin: 0.25rem 0;
        }
        
        @media (max-width: 768px) {
            .container {
                margin: 1rem;
                border-radius: 5px;
            }
            
            .header h1 {
                font-size: 2rem;
            }
            
            .report-info {
                flex-direction: column;
                gap: 1rem;
            }
            
            section {
                padding: 1rem;
            }
            
            .plots-grid {
                grid-template-columns: 1fr;
            }
        }
        """
    
    def _generate_pdf_html(self):
        """Generate simplified HTML for PDF conversion"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>DataViz EDA Report - {self.filename}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; text-align: center; }}
                h2 {{ color: #34495e; border-bottom: 2px solid #3498db; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>🧠 DataViz - EDA Report</h1>
            <div class="summary">
                <h2>Dataset Summary</h2>
                <p><strong>Filename:</strong> {self.filename}</p>
                <p><strong>Generated:</strong> {self.generated_time}</p>
                <p><strong>Shape:</strong> {len(self.df):,} rows × {len(self.df.columns)} columns</p>
                <p><strong>Missing Values:</strong> {self.df.isnull().sum().sum():,}</p>
            </div>
            
            <h2>Statistical Summary</h2>
            {self.df.describe().to_html() if len(self.numeric_cols) > 0 else '<p>No numerical columns found.</p>'}
            
            <h2>Missing Values Analysis</h2>
            {self._get_missing_values_table()}
            
            <h2>Data Types</h2>
            {pd.DataFrame({'Column': self.df.columns, 'Type': self.df.dtypes.astype(str), 'Non-Null': self.df.count()}).to_html(index=False)}
        </body>
        </html>
        """
    
    def _get_missing_values_table(self):
        """Get missing values table for PDF"""
        missing_counts = self.df.isnull().sum()
        missing_percentages = (missing_counts / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing_counts.index,
            'Missing Count': missing_counts.values,
            'Missing %': missing_percentages.round(2)
        })
        
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False)
        
        if missing_df.empty:
            return "<p>🎉 No missing values found in the dataset!</p>"
        
        return missing_df.to_html(index=False)
    
    def _generate_simple_pdf(self, analyzer):
        """Fallback simple PDF generation using FPDF"""
        try:
            from fpdf import FPDF
            
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font('Arial', 'B', 16)
            
            # Title
            pdf.cell(0, 10, f'DataViz EDA Report - {self.filename}', 0, 1, 'C')
            pdf.ln(10)
            
            # Basic info
            pdf.set_font('Arial', '', 12)
            pdf.cell(0, 10, f'Generated: {self.generated_time}', 0, 1)
            pdf.cell(0, 10, f'Dataset Shape: {len(self.df):,} rows x {len(self.df.columns)} columns', 0, 1)
            pdf.cell(0, 10, f'Missing Values: {self.df.isnull().sum().sum():,}', 0, 1)
            pdf.ln(10)
            
            # Add more content as needed
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, 'Summary Statistics', 0, 1)
            pdf.set_font('Arial', '', 10)
            
            if self.numeric_cols:
                stats = self.df[self.numeric_cols].describe()
                for col in stats.columns[:3]:  # Limit columns for space
                    pdf.cell(0, 8, f'{col}: Mean={stats.loc["mean", col]:.2f}, Std={stats.loc["std", col]:.2f}', 0, 1)
            
            return pdf.output(dest='S').encode('latin-1')
            
        except ImportError:
            # If FPDF is not available, return a simple text report
            report_text = f"""
            DataViz EDA Report
            ==================
            
            Dataset: {self.filename}
            Generated: {self.generated_time}
            Shape: {len(self.df):,} rows × {len(self.df.columns)} columns
            Missing Values: {self.df.isnull().sum().sum():,}
            
            Column Types:
            - Numerical: {len(self.numeric_cols)}
            - Categorical: {len(self.categorical_cols)}
            - DateTime: {len(self.datetime_cols)}
            - Boolean: {len(self.boolean_cols)}
            
            For detailed analysis, please use the HTML report.
            """
            
            return report_text.encode('utf-8')
