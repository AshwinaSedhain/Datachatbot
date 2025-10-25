"""
Domain-Aware Complete Auto-Visualization Engine
Supports 17+ chart types with intelligent domain-based selection
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Tuple, Optional

class AutoVisualizer:
    """Complete visualization suite with domain awareness"""
    
    def __init__(self):
        # Domain-specific chart preferences
        self.domain_chart_preferences = {
            'healthcare': {
                'primary': ['bar', 'line', 'pie', 'funnel'],
                'avoid': ['treemap'],
                'colors': 'Blues',
                'metrics_focus': ['patient_count', 'diagnosis_rate', 'treatment_outcome']
            },
            'finance': {
                'primary': ['waterfall', 'line', 'bar', 'heatmap'],
                'avoid': ['funnel'],
                'colors': 'Greens',
                'metrics_focus': ['revenue', 'profit', 'cost', 'margin']
            },
            'hospital': {
                'primary': ['bar', 'heatmap', 'gauge', 'line'],
                'avoid': [],
                'colors': 'Teal',
                'metrics_focus': ['bed_occupancy', 'patient_flow', 'department_load']
            },
            'retail': {
                'primary': ['bar', 'pie', 'line', 'treemap'],
                'avoid': ['waterfall'],
                'colors': 'Oranges',
                'metrics_focus': ['sales', 'inventory', 'customer_count']
            },
            'education': {
                'primary': ['bar', 'line', 'histogram', 'box'],
                'avoid': ['waterfall'],
                'colors': 'Purples',
                'metrics_focus': ['enrollment', 'grades', 'attendance']
            },
            'hr': {
                'primary': ['bar', 'pie', 'heatmap', 'funnel'],
                'avoid': [],
                'colors': 'Blues',
                'metrics_focus': ['headcount', 'turnover', 'salary']
            },
            'logistics': {
                'primary': ['line', 'heatmap', 'bar', 'sankey'],
                'avoid': ['pie'],
                'colors': 'Reds',
                'metrics_focus': ['delivery_time', 'routes', 'capacity']
            },
            'ecommerce': {
                'primary': ['funnel', 'bar', 'line', 'treemap'],
                'avoid': [],
                'colors': 'Viridis',
                'metrics_focus': ['conversion', 'cart_value', 'sessions']
            }
        }
    
    def create_chart(
        self,
        data: pd.DataFrame,
        question: str,
        intent: str,
        domain: str = 'general'
    ) -> Tuple[Optional[go.Figure], str]:
        """
        Intelligently select and create the best visualization with domain awareness
        """
        
        if data is None or data.empty:
            return None, "none"
        
        # Get domain preferences
        domain_pref = self.domain_chart_preferences.get(
            domain, 
            {'primary': [], 'avoid': [], 'colors': 'Blues', 'metrics_focus': []}
        )
        
        # Analyze data structure
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        date_cols = data.select_dtypes(include=['datetime64']).columns.tolist()
        
        question_lower = question.lower()
        
        # Get domain color scheme
        color_scheme = domain_pref.get('colors', 'Blues')
        
        # Chart selection logic with domain awareness
        
        # 1. FINANCIAL ANALYSIS (Waterfall)
        if domain == 'finance' and any(word in question_lower for word in ['profit', 'loss', 'breakdown', 'p&l']):
            if len(data) <= 10 and numeric_cols:
                return self._create_waterfall(data, categorical_cols[0] if categorical_cols else None, numeric_cols[0], color_scheme), "waterfall"
        
        # 2. CONVERSION FUNNEL
        if any(word in question_lower for word in ['funnel', 'conversion', 'pipeline', 'stages']):
            if categorical_cols and numeric_cols and 'funnel' not in domain_pref.get('avoid', []):
                return self._create_funnel(data, categorical_cols[0], numeric_cols[0], color_scheme), "funnel"
        
        # 3. HIERARCHY (Treemap)
        if any(word in question_lower for word in ['hierarchy', 'breakdown', 'composition']) and len(categorical_cols) >= 2:
            if 'treemap' not in domain_pref.get('avoid', []):
                return self._create_treemap(data, categorical_cols[:2], numeric_cols[0] if numeric_cols else None, color_scheme), "treemap"
        
        # 4. DISTRIBUTION
        if any(word in question_lower for word in ['distribution', 'spread', 'range']):
            if numeric_cols:
                if domain in ['education', 'healthcare']:
                    return self._create_box_plot(data, numeric_cols[0], color_scheme), "box"
                else:
                    return self._create_histogram(data, numeric_cols[0], color_scheme), "histogram"
        
        # 5. CORRELATION
        if any(word in question_lower for word in ['correlation', 'relationship', 'vs', 'versus']):
            if len(numeric_cols) >= 2:
                return self._create_scatter(data, numeric_cols[0], numeric_cols[1], color_scheme), "scatter"
        
        # 6. COMPARISON
        if any(word in question_lower for word in ['compare', 'comparison']) and len(categorical_cols) >= 2:
            return self._create_grouped_bar(data, categorical_cols[0], categorical_cols[1], numeric_cols[0], color_scheme), "grouped_bar"
        
        # 7. TREND
        if date_cols and numeric_cols:
            return self._create_line_chart(data, date_cols[0], numeric_cols[0], color_scheme, domain), "line"
        
        # 8. COMPOSITION
        if any(word in question_lower for word in ['share', 'percentage', 'proportion']) and len(data) <= 10:
            if categorical_cols and numeric_cols and 'pie' not in domain_pref.get('avoid', []):
                return self._create_pie_chart(data, categorical_cols[0], numeric_cols[0], color_scheme), "pie"
        
        # 9. TOP/BOTTOM
        if intent == 'top_bottom' or any(word in question_lower for word in ['top', 'bottom', 'best', 'worst']):
            if categorical_cols and numeric_cols:
                return self._create_bar_chart(data, categorical_cols[0], numeric_cols[0], color_scheme, domain), "bar"
        
        # 10. HEATMAP
        if any(word in question_lower for word in ['pattern', 'heatmap', 'matrix']):
            if len(categorical_cols) >= 2 and numeric_cols:
                return self._create_heatmap(data, categorical_cols[0], categorical_cols[1], numeric_cols[0], color_scheme), "heatmap"
        
        # DEFAULT
        if len(data) > 100:
            return self._create_table(data.head(100), domain), "table"
        elif categorical_cols and numeric_cols:
            return self._create_bar_chart(data, categorical_cols[0], numeric_cols[0], color_scheme, domain), "bar"
        else:
            return self._create_table(data, domain), "table"
    
    # ========================================
    # CHART IMPLEMENTATIONS
    # ========================================
    
    def _create_bar_chart(self, data: pd.DataFrame, x_col: str, y_col: str, color_scheme: str, domain: str = 'general') -> go.Figure:
        """1. Bar Chart"""
        data_sorted = data.nlargest(20, y_col) if len(data) > 20 else data
        
        fig = px.bar(
            data_sorted,
            x=x_col,
            y=y_col,
            title=f"{y_col.replace('_', ' ').title()} by {x_col.replace('_', ' ').title()} ({domain.title()})",
            color=y_col,
            color_continuous_scale=color_scheme,
            text=y_col
        )
        fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        fig.update_layout(showlegend=False, xaxis_tickangle=-45)
        return fig
    
    def _create_pie_chart(self, data: pd.DataFrame, labels_col: str, values_col: str, color_scheme: str) -> go.Figure:
        """2. Pie/Donut Chart"""
        if len(data) > 8:
            top_data = data.nlargest(8, values_col)
            others_sum = data.nsmallest(len(data)-8, values_col)[values_col].sum()
            others_row = pd.DataFrame({labels_col: ['Others'], values_col: [others_sum]})
            data = pd.concat([top_data, others_row], ignore_index=True)
        
        fig = px.pie(
            data,
            names=labels_col,
            values=values_col,
            title=f"{values_col.replace('_', ' ').title()} Distribution",
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        return fig
    
    def _create_line_chart(self, data: pd.DataFrame, x_col: str, y_col: str, color_scheme: str, domain: str = 'general') -> go.Figure:
        """3. Line Chart"""
        fig = px.line(
            data,
            x=x_col,
            y=y_col,
            title=f"{y_col.replace('_', ' ').title()} Trend Over Time ({domain.title()})",
            markers=True,
            line_shape='spline'
        )
        
        if domain == 'finance':
            fig.update_traces(fill='tozeroy', line=dict(width=3, color='green'))
        else:
            fig.update_traces(line=dict(width=3))
        
        return fig
    
    def _create_scatter(self, data: pd.DataFrame, x_col: str, y_col: str, color_scheme: str, size_col: str = None) -> go.Figure:
        """4. Scatter Plot"""
        fig = px.scatter(
            data,
            x=x_col,
            y=y_col,
            size=size_col,
            title=f"{y_col} vs {x_col}",
            trendline="ols"
        )
        return fig
    
    def _create_histogram(self, data: pd.DataFrame, col: str, color_scheme: str) -> go.Figure:
        """5. Histogram"""
        fig = px.histogram(
            data,
            x=col,
            title=f"{col.replace('_', ' ').title()} Distribution",
            nbins=30,
            marginal="box"
        )
        return fig
    
    def _create_box_plot(self, data: pd.DataFrame, col: str, color_scheme: str) -> go.Figure:
        """5b. Box Plot"""
        fig = px.box(
            data,
            y=col,
            title=f"{col.replace('_', ' ').title()} Distribution Analysis"
        )
        return fig
    
    def _create_grouped_bar(self, data: pd.DataFrame, x_col: str, color_col: str, y_col: str, color_scheme: str) -> go.Figure:
        """6. Grouped Bar Chart"""
        fig = px.bar(
            data,
            x=x_col,
            y=y_col,
            color=color_col,
            title=f"{y_col} by {x_col} and {color_col}",
            barmode='group'
        )
        return fig
    
    def _create_heatmap(self, data: pd.DataFrame, x_col: str, y_col: str, value_col: str, color_scheme: str) -> go.Figure:
        """7. Heatmap"""
        pivot = data.pivot_table(values=value_col, index=y_col, columns=x_col, aggfunc='sum', fill_value=0)
        
        fig = px.imshow(
            pivot,
            title=f"{value_col} Heatmap",
            color_continuous_scale=color_scheme,
            aspect='auto'
        )
        return fig
    
    def _create_treemap(self, data: pd.DataFrame, path_cols: list, value_col: str, color_scheme: str) -> go.Figure:
        """8. Treemap"""
        fig = px.treemap(
            data,
            path=path_cols,
            values=value_col,
            title=f"{value_col} Hierarchy",
            color=value_col,
            color_continuous_scale=color_scheme
        )
        return fig
    
    def _create_waterfall(self, data: pd.DataFrame, x_col: str, y_col: str, color_scheme: str) -> go.Figure:
        """9. Waterfall Chart"""
        if x_col is None or x_col not in data.columns:
            x_col = data.columns[0]
        
        measure = ['relative'] * len(data)
        if len(data) > 0:
            measure[0] = 'absolute'
        if len(data) > 1:
            measure[-1] = 'total'
        
        fig = go.Figure(go.Waterfall(
            name="Financial Breakdown",
            orientation="v",
            measure=measure,
            x=data[x_col],
            y=data[y_col],
            text=data[y_col],
            textposition="outside",
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "green"}},
            decreasing={"marker": {"color": "red"}},
            totals={"marker": {"color": "blue"}}
        ))
        fig.update_layout(title="Financial Waterfall Analysis")
        return fig
    
    def _create_funnel(self, data: pd.DataFrame, stage_col: str, value_col: str, color_scheme: str) -> go.Figure:
        """10. Funnel Chart"""
        fig = px.funnel(
            data,
            x=value_col,
            y=stage_col,
            title="Conversion Funnel"
        )
        return fig
    
    def _create_table(self, data: pd.DataFrame, domain: str = 'general') -> go.Figure:
        """11. Interactive Table"""
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(data.columns),
                fill_color='#1f77b4',
                align='left',
                font=dict(color='white', size=12)
            ),
            cells=dict(
                values=[data[col] for col in data.columns],
                fill_color='lavender',
                align='left'
            )
        )])
        fig.update_layout(
            title=f"Data Results ({domain.title()} Domain)", 
            height=min(600, len(data)*30 + 100)
        )
        return fig