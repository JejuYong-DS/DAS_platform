from shiny import App, ui, render, reactive
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO, BytesIO
import base64

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression, SGDClassifier, SGDRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
import scipy.stats as stats

try:
    from bayes_opt import BayesianOptimization
    BAYESOPT_AVAILABLE = True
except ImportError:
    BAYESOPT_AVAILABLE = False

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

app_ui = ui.page_fluid(
    ui.panel_title("Data Analysis Software (DAS)"),
    
    ui.navset_tab(
        ui.nav_panel(
            "Data Upload",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_file("file", "Upload File", accept=[".csv"], multiple=False),
                    ui.input_select("target_col", "Select Target Column", choices=[]),
                    ui.input_radio_buttons(
                        "target_type", 
                        "Target Type",
                        choices={"categorical": "Classification", "continuous": "Regression"}
                    ),
                    width=300
                ),
                ui.card(
                    ui.card_header("Data Preview"),
                    ui.div(
                        ui.output_data_frame("data_preview"),
                        style="height: 300px; overflow: auto;"
                    )
                ),
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Data Info"),
                        ui.div(
                            ui.output_text_verbatim("data_info"),
                            style="height: 300px; overflow-y: auto;"
                        )
                    ),
                    ui.card(
                        ui.card_header("Data Description"),
                        ui.div(
                            ui.output_table("data_describe"),
                            style="height: 300px; overflow: auto;"
                        )
                    ),
                    col_widths=[6, 6]
                )
            )
        ),
       
        ui.nav_panel(
            "Preprocessing",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.div(
                        ui.h5("Delete Variable", style="margin-bottom: 5px; margin-top: 0;"),
                        ui.input_select("delete_col", "Column", choices=[]),
                        ui.input_action_button("apply_delete", "Delete Variable", class_="btn-danger", style="margin-bottom: 0;"),
                        style="margin-bottom: 10px; padding-bottom: 10px; border-bottom: 1px solid #dee2e6;"
                    ),
                    
                    ui.div(
                        ui.h5("Outlier Removal", style="margin-bottom: 5px; margin-top: 0;"),
                        ui.input_select("outlier_col", "Column", choices=[]),
                        ui.input_select(
                            "outlier_method",
                            "Method",
                            choices={
                                "IQR": "IQR (Interquartile Range)",
                                "zscore": "Z-Score (threshold=3)"
                            }
                        ),
                        ui.input_action_button("apply_outlier", "Remove Outliers", class_="btn-danger", style="margin-bottom: 0;"),
                        style="margin-bottom: 10px; padding-bottom: 10px; border-bottom: 1px solid #dee2e6;"
                    ),
                    
                    ui.div(
                        ui.h5("Missing Values Handling", style="margin-bottom: 5px; margin-top: 0;"),
                        ui.input_select("missing_col", "Column", choices=[]),
                        ui.input_select(
                            "missing_method",
                            "Method",
                            choices={
                                "drop": "Drop",
                                "mean": "Mean",
                                "median": "Median",
                                "mode": "Mode",
                                "ffill": "Forward Fill",
                                "bfill": "Backward Fill",
                                "linear_interpolate": "Linear Interpolation",
                                "cubic_spline": "Cubic Spline Interpolation"
                            }
                        ),
                        ui.input_action_button("apply_missing", "Apply", class_="btn-primary", style="margin-bottom: 0;"),
                        style="margin-bottom: 10px; padding-bottom: 10px; border-bottom: 1px solid #dee2e6;"
                    ),
                    
                    ui.div(
                        ui.h5("Label Encoding", style="margin-bottom: 5px; margin-top: 0;"),
                        ui.input_select("encode_col", "Column", choices=[]),
                        ui.input_action_button("apply_encode", "Apply Encoding", class_="btn-primary", style="margin-bottom: 0;"),
                        style="margin-bottom: 10px; padding-bottom: 10px; border-bottom: 1px solid #dee2e6;"
                    ),
                    
                    ui.div(
                        ui.h5("Scaling", style="margin-bottom: 5px; margin-top: 0;"),
                        ui.input_select(
                            "scale_method",
                            "Method",
                            choices={
                                "standard": "Standard",
                                "minmax": "Min Max",
                                "robust": "Robust"
                            }
                        ),
                        ui.input_action_button("apply_scale", "Apply Scaling", class_="btn-primary", style="margin-bottom: 0;"),
                        style="margin-bottom: 10px; padding-bottom: 10px; border-bottom: 1px solid #dee2e6;"
                    ),
                    
                    ui.div(
                        ui.h5("Export Data", style="margin-bottom: 5px; margin-top: 0;"),
                        ui.download_button("download_processed_data", "Download Processed CSV", 
                                         style="width: 100%; background-color: #28a745; color: white; border: none;"),
                        style="margin-bottom: 10px;"
                    ),
                    width=300,
                    style="max-height: 90vh; overflow-y: auto; padding: 10px;"
                ),
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Processed Data"),
                        ui.div(
                            ui.output_data_frame("processed_data"),
                            style="height: 500px; overflow: auto;"
                        )
                    ),
                    ui.card(
                        ui.card_header("Missing Values Summary"),
                        ui.div(
                            ui.output_table("missing_summary"),
                            style="height: 500px; overflow-y: auto;"
                        )
                    ),
                    col_widths=[9, 3]
                )
            )
        ),
        
        ui.nav_panel(
            "Visualization",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_select("viz_type", "Visualization Method", 
                                  choices={
                                      "histogram": "Histogram",
                                      "boxplot": "Box Plot",
                                      "barplot": "Bar Plot",
                                      "pieplot": "Pie Chart",
                                      "scatterplot": "Scatter Plot",
                                      "lineplot": "Line Plot",
                                      "correlation": "Correlation Heatmap"
                                  }),
                    ui.input_action_button("set_viz_variables", "Set Variables", style="width: 100%; margin-bottom: 10px; background-color: #9ACD32; color: white; border: none;"),
                    ui.input_action_button("generate_viz", "Generate", class_="btn-success", style="width: 100%;"),
                    width=300
                ),
                ui.output_plot("visualization", height="800px")
            )
        ),
        
        ui.nav_panel(
            "Statistical Analysis",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.div(
                        ui.div(
                            ui.span("Statistical Test", style="font-weight: 500;"),
                            ui.input_action_button("show_stat_info", "ⓘ", style="padding: 0px 6px; font-size: 14px; margin-left: 5px; border: none; background: transparent; color: #007bff; cursor: pointer;"),
                            style="display: flex; align-items: center; margin-bottom: 5px;"
                        ),
                        ui.input_select(
                            "stat_test",
                            label=None,
                            choices={
                                "ttest": "T-Test (Independent)",
                                "ttest_paired": "T-Test (Paired)",
                                "anova": "One-Way ANOVA",
                                "anova_two": "Two-Way ANOVA",
                                "chi_square": "Chi-Square Test",
                                "mannwhitney": "Mann-Whitney U Test",
                                "kruskal": "Kruskal-Wallis H Test",
                                "shapiro": "Shapiro-Wilk (Normality)",
                                "cronbach": "Cronbach's Alpha",
                                "bartlett": "Bartlett's Test of Sphericity",
                                "kmo": "KMO Test"
                            }
                        ),
                    ),
                    ui.input_action_button("set_stat_columns", "Set Variables", style="width: 100%; margin-bottom: 10px; background-color: #9ACD32; color: white; border: none;"),
                    ui.input_action_button("run_stat", "Run Test", class_="btn-success", style="width: 100%;"),
                    width=300
                ),
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Test Results"),
                        ui.output_text_verbatim("stat_results")
                    ),
                    col_widths=[12]
                ),
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Group Statistics"),
                        ui.div(
                            ui.output_data_frame("group_stats"),
                            style="height: 300px; overflow: auto;"
                        )
                    ),
                    col_widths=[12]
                )
            )
        ),
        
        ui.nav_panel(
            "Dimensionality Reduction",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.div(
                        ui.div(
                            ui.span("Method", style="font-weight: 500;"),
                            ui.input_action_button("show_dim_info", "ⓘ", style="padding: 0px 6px; font-size: 14px; margin-left: 5px; border: none; background: transparent; color: #007bff; cursor: pointer;"),
                            style="display: flex; align-items: center; margin-bottom: 5px;"
                        ),
                        ui.input_select(
                            "dim_method",
                            label=None,
                            choices={
                                "pca": "Principal Component Analysis (PCA)",
                                "fa": "Factor Analysis (FA)"
                            }
                        ),
                    ),
                    ui.input_action_button("set_dim_settings", "Settings", style="width: 100%; margin-bottom: 10px; background-color: #9ACD32; color: white; border: none;"),
                    ui.input_action_button("run_dim_reduction", "Run Analysis", class_="btn-success", style="width: 100%;"),
                    width=300
                ),
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Scree Plot"),
                        ui.output_plot("dim_scree_plot", height="400px")
                    ),
                    col_widths=[12]
                ),
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Loadings"),
                        ui.div(
                            ui.output_data_frame("dim_loadings"),
                            style="height: 400px; overflow: auto;"
                        )
                    ),
                    ui.card(
                        ui.card_header("Explained Variance"),
                        ui.div(
                            ui.output_data_frame("dim_variance"),
                            style="height: 400px; overflow: auto;"
                        )
                    ),
                    col_widths=[6, 6]
                )
            )
        ),
        
        ui.nav_panel(
            "Modeling",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.div(
                        ui.div(
                            ui.span("Select Model", style="font-weight: 500;"),
                            ui.input_action_button("show_model_info", "ⓘ", style="padding: 0px 6px; font-size: 14px; margin-left: 5px; border: none; background: transparent; color: #007bff; cursor: pointer;"),
                            style="display: flex; align-items: center; margin-bottom: 5px;"
                        ),
                        ui.input_select(
                            "model_name",
                            label=None,
                            choices={
                                "linear regression": "Linear Regression",
                                "ridge": "Ridge",
                                "lasso": "Lasso",
                                "logistic regression": "Logistic Regression",
                                "support vector machine": "Support Vector Machine",
                                "decision tree": "Decision Tree",
                                "random forest": "Random Forest",
                                "stochastic gradient descent": "Stochastic Gradient Descent",
                                "gradient boosting": "Gradient Boosting",
                                "xgboost": "XGBoost",
                                "gaussian process": "Gaussian Process",
                                "lgbm": "LightGBM"
                            }
                        ),
                    ),
                    ui.input_numeric("random_state", "Random State", value=42),
                    ui.input_slider("test_size", "Test Size", min=0.1, max=0.5, value=0.2, step=0.05),
                    ui.input_action_button("set_model_variables", "Set Variables", style="width: 100%; margin-bottom: 10px; background-color: #9ACD32; color: white; border: none;"),
                    ui.input_action_button("set_params", "Set Parameters", style="width: 100%; margin-bottom: 10px; background-color: #9ACD32; color: white; border: none;"),
                    ui.input_action_button("auto_optimize", "Auto Optimize", style="width: 100%; margin-bottom: 10px; background-color: #FF8C00; color: white; border: none;"),
                    ui.input_action_button("train_model", "Train Model", class_="btn-success", style="width: 100%;"),
                    width=300
                ),
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Model Results"),
                        ui.output_text_verbatim("model_results")
                    ),
                    ui.card(
                        ui.card_header("Performance Metrics"),
                        ui.output_table("metrics_table")
                    ),
                    col_widths=[6, 6]
                ),
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Model Visualization"),
                        ui.output_plot("model_viz", height="400px")
                    ),
                    ui.card(
                        ui.card_header("Feature Importance"),
                        ui.output_plot("feature_importance", height="400px")
                    ),
                    col_widths=[6, 6]
                )
            )
        )
    )
)

def server(input, output, session):
    data = reactive.Value(None)
    original_data = reactive.Value(None)
    preprocessor = reactive.Value(None)
    trained_model = reactive.Value(None)
    encoding_info = reactive.Value({})
    model_params = reactive.Value({})
    stat_columns = reactive.Value({'group': None, 'value': None})
    viz_variables = reactive.Value({'x': None, 'y': None})
    dim_settings = reactive.Value({'n_components': 2, 'rotation': 'varimax', 'variables': None})
    dim_results = reactive.Value(None)
    model_variables = reactive.Value(None)  # 독립변수 선택을 위한 reactive value
    
    @reactive.Effect
    @reactive.event(input.file)
    def load_data():
        file_info = input.file()
        if file_info is None:
            return
        
        df = pd.read_csv(file_info[0]["datapath"])
        data.set(df)
        original_data.set(df.copy())
        encoding_info.set({})
        
        cols = df.columns.tolist()
        ui.update_select("target_col", choices=cols)
        ui.update_select("delete_col", choices=cols)
        ui.update_select("missing_col", choices=cols)
        ui.update_select("outlier_col", choices=cols)
        ui.update_select("encode_col", choices=cols)
    
    @reactive.Effect
    @reactive.event(input.set_viz_variables)
    def show_viz_variables_modal():
        df = data.get()
        if df is None:
            ui.notification_show("Please load data first", type="error")
            return
        
        viz_type = input.viz_type()
        cols = df.columns.tolist()
        
        col_choices = {"": "(Not Selected)"}
        for col in cols:
            col_choices[col] = col
        
        current_vars = viz_variables.get()
        
        # Determine what inputs to show based on visualization type
        if viz_type == "correlation":
            # Only show exclude target option
            m = ui.modal(
                ui.div(
                    ui.h5("Correlation Heatmap Settings", style="margin-bottom: 15px;"),
                    ui.input_checkbox("viz_exclude_target", "Exclude Target Variable", value=current_vars.get('exclude_target', False)),
                    ui.p("If checked, the target variable will be excluded from the correlation matrix.", 
                         style="font-size: 0.85em; color: #666; margin-top: 5px;"),
                ),
                title="Variable Selection",
                size="m",
                easy_close=True,
                footer=ui.div(
                    ui.input_action_button("save_viz_variables", "Save", class_="btn-primary"),
                    ui.input_action_button("cancel_viz_variables", "Cancel", class_="btn-secondary"),
                    style="display: flex; gap: 10px; justify-content: flex-end;"
                )
            )
        elif viz_type in ["scatterplot", "lineplot", "heatmap"]:
            # Need X and Y variables
            m = ui.modal(
                ui.div(
                    ui.h5(f"Set Variables for {viz_type.title()}", style="margin-bottom: 15px;"),
                    ui.div(
                        ui.input_select("viz_x_col", "X Variable", choices=col_choices, 
                                      selected=current_vars.get('x', "")),
                        ui.p("Select the variable for X-axis", style="font-size: 0.85em; color: #666; margin-top: 5px;"),
                        style="margin-bottom: 15px;"
                    ),
                    ui.div(
                        ui.input_select("viz_y_col", "Y Variable", choices=col_choices, 
                                      selected=current_vars.get('y', "")),
                        ui.p("Select the variable for Y-axis", style="font-size: 0.85em; color: #666; margin-top: 5px;"),
                    ),
                ),
                title="Variable Selection",
                size="m",
                easy_close=True,
                footer=ui.div(
                    ui.input_action_button("save_viz_variables", "Save", class_="btn-primary"),
                    ui.input_action_button("cancel_viz_variables", "Cancel", class_="btn-secondary"),
                    style="display: flex; gap: 10px; justify-content: flex-end;"
                )
            )
        else:
            # histogram, boxplot, barplot, pieplot - need X only
            m = ui.modal(
                ui.div(
                    ui.h5(f"Set Variables for {viz_type.title()}", style="margin-bottom: 15px;"),
                    ui.div(
                        ui.input_select("viz_x_col", "Variable", choices=col_choices, 
                                      selected=current_vars.get('x', "")),
                        ui.p("Select the variable to visualize", style="font-size: 0.85em; color: #666; margin-top: 5px;"),
                    ),
                ),
                title="Variable Selection",
                size="m",
                easy_close=True,
                footer=ui.div(
                    ui.input_action_button("save_viz_variables", "Save", class_="btn-primary"),
                    ui.input_action_button("cancel_viz_variables", "Cancel", class_="btn-secondary"),
                    style="display: flex; gap: 10px; justify-content: flex-end;"
                )
            )
        
        ui.modal_show(m)
    
    @reactive.Effect
    @reactive.event(input.save_viz_variables)
    def save_viz_variables():
        try:
            viz_type = input.viz_type()
            
            if viz_type == "correlation":
                exclude_target = input.viz_exclude_target()
                viz_variables.set({'x': None, 'y': None, 'exclude_target': exclude_target})
                ui.modal_remove()
                ui.notification_show(f"Settings saved: Exclude target = {exclude_target}", type="message")
            else:
                x_col = input.viz_x_col() if input.viz_x_col() else None
                y_col = input.viz_y_col() if viz_type in ["scatterplot", "lineplot"] and input.viz_y_col() else None
                
                viz_variables.set({'x': x_col, 'y': y_col})
                ui.modal_remove()
                
                var_info = f"X={x_col or 'None'}"
                if viz_type in ["scatterplot", "lineplot"]:
                    var_info += f", Y={y_col or 'None'}"
                
                ui.notification_show(f"Variables saved: {var_info}", type="message")
        except Exception as e:
            ui.notification_show(f"Error saving variables: {str(e)}", type="error")
    
    @reactive.Effect
    @reactive.event(input.cancel_viz_variables)
    def cancel_viz_variables():
        ui.modal_remove()
    
    @output
    @render.data_frame
    def data_preview():
        df = original_data.get()
        if df is not None:
            return render.DataGrid(df, width="100%", row_selection_mode="none")
        return None
    
    @output
    @render.text
    def data_info():
        df = original_data.get()
        if df is not None:
            buffer = StringIO()
            df.info(buf=buffer)
            return buffer.getvalue()
        return "No data loaded"
    
    @output
    @render.table
    def data_describe():
        df = original_data.get()
        if df is not None:
            return df.describe().reset_index()
        return pd.DataFrame()
    
    @output
    @render.table
    def missing_summary():
        df = data.get()
        if df is not None:
            missing = df.isnull().sum()
            missing_pct = (missing / len(df) * 100).round(2)
            summary = pd.DataFrame({
                'Column': missing.index,
                'Missing Count': missing.values,
                'Percentage': missing_pct.values
            })
            return summary[summary['Missing Count'] > 0]
        return pd.DataFrame()
    
    @reactive.Effect
    @reactive.event(input.apply_delete)
    def handle_delete():
        df = data.get()
        if df is None:
            return
        
        col = input.delete_col()
        
        if not col:
            ui.notification_show("Please select a column to delete", type="error")
            return
        
        df = df.copy()
        df = df.drop(columns=[col])
        data.set(df)
        
        cols = df.columns.tolist()
        ui.update_select("target_col", choices=cols)
        ui.update_select("delete_col", choices=cols)
        ui.update_select("missing_col", choices=cols)
        ui.update_select("outlier_col", choices=cols)
        ui.update_select("encode_col", choices=cols)
        
        ui.notification_show(f"Column '{col}' deleted successfully", type="message")
    
    @reactive.Effect
    @reactive.event(input.apply_missing)
    def handle_missing():
        df = data.get()
        if df is None:
            return
        
        col = input.missing_col()
        method = input.missing_method()
        
        df = df.copy()
        
        if method == "drop":
            df = df.dropna(subset=[col])
        elif method == "mean":
            df[col] = df[col].fillna(df[col].mean())
        elif method == "median":
            df[col] = df[col].fillna(df[col].median())
        elif method == "mode":
            if not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode().iloc[0])
        elif method == "ffill":
            df[col] = df[col].fillna(method='ffill')
        elif method == "bfill":
            df[col] = df[col].fillna(method='bfill')
        elif method == "linear_interpolate":
            df[col] = df[col].interpolate(method="linear")
        elif method == "cubic_spline":
            df[col] = df[col].interpolate(method="spline", order=3)
        
        data.set(df)
        ui.notification_show(f"Missing values handled for {col} using {method}", type="message")
    
    @reactive.Effect
    @reactive.event(input.apply_outlier)
    def handle_outlier():
        df = data.get()
        if df is None:
            return
        
        col = input.outlier_col()
        method = input.outlier_method()
        
        if not col:
            ui.notification_show("Please select a column", type="error")
            return
        
        if not pd.api.types.is_numeric_dtype(df[col]):
            ui.notification_show("Outlier removal only works with numeric columns", type="error")
            return
        
        df = df.copy()
        original_count = len(df)
        
        try:
            if method == "IQR":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                
            elif method == "zscore":
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                threshold = 3
                valid_indices = df[col].notna()
                valid_indices.loc[df[col].notna()] = z_scores <= threshold
                df = df[valid_indices]
            
            removed_count = original_count - len(df)
            data.set(df)
            
            ui.notification_show(
                f"Removed {removed_count} outliers from '{col}' using {method} method",
                type="message"
            )
            
        except Exception as e:
            ui.notification_show(f"Error removing outliers: {str(e)}", type="error")
    
    @reactive.Effect
    @reactive.event(input.apply_encode)
    def handle_encoding():
        df = data.get()
        if df is None:
            return
        
        col = input.encode_col()
        df = df.copy()
        
        encoder = LabelEncoder()
        original_classes = df[col].unique()
        df[col] = encoder.fit_transform(df[col].astype(str))
        
        enc_info = encoding_info.get()
        enc_info[col] = {
            'classes': encoder.classes_.tolist(),
            'mapping': {str(cls): int(idx) for idx, cls in enumerate(encoder.classes_)}
        }
        encoding_info.set(enc_info)
        
        data.set(df)
        ui.notification_show(f"Encoding applied to {col}\n{enc_info[col]['mapping']}", type="message")
    
    @reactive.Effect
    @reactive.event(input.apply_scale)
    def handle_scaling():
        df = data.get()
        if df is None:
            return
        
        target_col = input.target_col()
        method = input.scale_method()
        
        df = df.copy()
        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]
        
        if not feature_cols:
            ui.notification_show("No numeric columns to scale", type="warning")
            return
        
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        else:
            scaler = RobustScaler()
        
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        data.set(df)
        ui.notification_show(f"Scaling applied using {method} method", type="message")
    
    @output
    @render.data_frame
    def processed_data():
        df = data.get()
        if df is not None:
            return render.DataGrid(df, width="100%", row_selection_mode="none")
        return None
    
    @session.download(filename="processed_data.csv")
    def download_processed_data():
        df = data.get()
        if df is None:
            yield ""
            return
        
        # Convert dataframe to CSV and yield as bytes
        csv_string = df.to_csv(index=False)
        yield csv_string
    
    @output
    @render.plot
    @reactive.event(input.generate_viz)
    def visualization():
        df = data.get()
        if df is None:
            return
        
        viz_type = input.viz_type()
        vars = viz_variables.get()
        x_col = vars.get('x')
        y_col = vars.get('y')
        exclude_target = vars.get('exclude_target', False)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        try:
            if viz_type == "histogram":
                if not x_col:
                    ax.text(0.5, 0.5, 'Please set variables first', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=12)
                elif x_col in df.columns:
                    df[x_col].dropna().hist(bins=30, ax=ax, edgecolor='black', color='steelblue')
                    ax.set_title(f"Histogram of {x_col}", fontsize=14, fontweight='bold')
                    ax.set_xlabel(x_col, fontsize=12)
                    ax.set_ylabel("Frequency", fontsize=12)
            
            elif viz_type == "boxplot":
                if not x_col:
                    ax.text(0.5, 0.5, 'Please set variables first', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=12)
                elif x_col in df.columns:
                    bp = ax.boxplot(df[x_col].dropna(), patch_artist=True)
                    bp['boxes'][0].set_facecolor('lightblue')
                    ax.set_title(f"Box Plot of {x_col}", fontsize=14, fontweight='bold')
                    ax.set_ylabel(x_col, fontsize=12)
            
            elif viz_type == "barplot":
                if not x_col:
                    ax.text(0.5, 0.5, 'Please set variables first', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=12)
                elif x_col in df.columns:
                    value_counts = df[x_col].value_counts().sort_index()
                    ax.bar(range(len(value_counts)), value_counts.values, color='steelblue', edgecolor='black')
                    ax.set_xticks(range(len(value_counts)))
                    ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
                    ax.set_title(f"Bar Plot of {x_col}", fontsize=14, fontweight='bold')
                    ax.set_xlabel(x_col, fontsize=12)
                    ax.set_ylabel("Count", fontsize=12)
            
            elif viz_type == "pieplot":
                if not x_col:
                    ax.text(0.5, 0.5, 'Please set variables first', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=12)
                elif x_col in df.columns:
                    value_counts = df[x_col].value_counts()
                    colors = plt.cm.Set3(range(len(value_counts)))
                    ax.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%', 
                          startangle=90, colors=colors)
                    ax.set_title(f"Pie Chart of {x_col}", fontsize=14, fontweight='bold')
            
            elif viz_type == "scatterplot":
                if not x_col or not y_col:
                    ax.text(0.5, 0.5, 'Please set both X and Y variables', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=12)
                elif x_col in df.columns and y_col in df.columns:
                    ax.scatter(df[x_col], df[y_col], alpha=0.6, color='steelblue', s=50)
                    ax.set_xlabel(x_col, fontsize=12)
                    ax.set_ylabel(y_col, fontsize=12)
                    ax.set_title(f"Scatter Plot: {x_col} vs {y_col}", fontsize=14, fontweight='bold')
            
            elif viz_type == "lineplot":
                if not x_col or not y_col:
                    ax.text(0.5, 0.5, 'Please set both X and Y variables', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=12)
                elif x_col in df.columns and y_col in df.columns:
                    df_sorted = df.sort_values(x_col)
                    ax.plot(df_sorted[x_col], df_sorted[y_col], marker='o', color='steelblue', linewidth=2)
                    ax.set_xlabel(x_col, fontsize=12)
                    ax.set_ylabel(y_col, fontsize=12)
                    ax.set_title(f"Line Plot: {x_col} vs {y_col}", fontsize=14, fontweight='bold')
            
            elif viz_type == "correlation":
                target_col = input.target_col() if exclude_target else None
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if target_col and target_col in numeric_cols:
                    numeric_cols = [col for col in numeric_cols if col != target_col]
                
                if len(numeric_cols) > 1:
                    corr = df[numeric_cols].corr()
                    sns.heatmap(corr, annot=True, cmap="RdBu_r", center=0, vmin=-1, vmax=1, 
                               ax=ax, fmt='.2f', square=True, linewidths=0.5, 
                               cbar_kws={'label': 'Correlation'})
                    title = "Correlation Heatmap"
                    if exclude_target and target_col:
                        title += f" (excluding {target_col})"
                    ax.set_title(title, fontsize=14, fontweight='bold')
                else:
                    ax.text(0.5, 0.5, 'Not enough numeric columns for correlation', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=12)
        
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12, color='red')
        
        plt.tight_layout()
        return fig
        
        plt.tight_layout()
        return fig
    
    @reactive.Effect
    @reactive.event(input.show_stat_info)
    def show_stat_info_modal():
        m = ui.modal(
            ui.div(
                ui.h5("Statistical Tests", style="font-weight: bold; margin-bottom: 15px; color: #2c3e50;"),
                ui.p("• T-Test (Independent): Compare means of two independent groups", style="font-size: 0.9em; margin-bottom: 8px; margin-left: 10px;"),
                ui.p("• T-Test (Paired): Compare means of two related/paired groups", style="font-size: 0.9em; margin-bottom: 8px; margin-left: 10px;"),
                ui.p("• One-Way ANOVA: Compare means of three or more independent groups", style="font-size: 0.9em; margin-bottom: 8px; margin-left: 10px;"),
                ui.p("• Two-Way ANOVA: Examine interaction between two independent variables", style="font-size: 0.9em; margin-bottom: 8px; margin-left: 10px;"),
                ui.p("• Chi-Square Test: Test independence between categorical variables", style="font-size: 0.9em; margin-bottom: 8px; margin-left: 10px;"),
                ui.p("• Mann-Whitney U Test: Non-parametric alternative to independent t-test", style="font-size: 0.9em; margin-bottom: 8px; margin-left: 10px;"),
                ui.p("• Kruskal-Wallis H Test: Non-parametric alternative to one-way ANOVA", style="font-size: 0.9em; margin-bottom: 8px; margin-left: 10px;"),
                ui.p("• Shapiro-Wilk Test: Test whether data follows normal distribution", style="font-size: 0.9em; margin-bottom: 8px; margin-left: 10px;"),
                ui.p("• Cronbach's Alpha: Measure internal consistency reliability of scale", style="font-size: 0.9em; margin-bottom: 8px; margin-left: 10px;"),
                ui.p("• Bartlett's Test: Test if variables are correlated (required for factor analysis)", style="font-size: 0.9em; margin-bottom: 8px; margin-left: 10px;"),
                ui.p("• KMO Test: Measure sampling adequacy for factor analysis", style="font-size: 0.9em; margin-bottom: 0; margin-left: 10px;"),
                style="padding: 20px;"
            ),
            title="Statistical Test Information",
            size="l",
            easy_close=True,
            footer=ui.div(
                ui.input_action_button("close_stat_info", "Close", class_="btn-secondary"),
                style="display: flex; justify-content: flex-end;"
            )
        )
        ui.modal_show(m)
    
    @reactive.Effect
    @reactive.event(input.close_stat_info)
    def close_stat_info_modal():
        ui.modal_remove()
    
    @reactive.Effect
    @reactive.event(input.set_stat_columns)
    def show_stat_columns_modal():
        df = data.get()
        if df is None:
            ui.notification_show("Please load data first", type="error")
            return
        
        stat_test = input.stat_test()
        cols = df.columns.tolist()
        
        col_choices = {"": "(Not Selected)"}
        for col in cols:
            col_choices[col] = col
        
        current_cols = stat_columns.get()
        
        if stat_test == "anova_two":
            group_label = "Group Variables (Select exactly 2)"
            value_label = "Dependent Variable"
            group_help = "Select exactly 2 independent variables for Two-Way ANOVA"
            value_help = "Select the dependent numeric variable"
            allow_multiple_group = True
            show_value = True
        elif stat_test in ["cronbach", "bartlett", "kmo"]:
            group_label = "Variables (Select multiple)"
            value_label = "Dependent Variable (not used)"
            group_help = "Select multiple variables to test reliability/adequacy"
            value_help = "This field is not used for this test"
            allow_multiple_group = True
            show_value = False
        elif stat_test in ["ttest", "ttest_paired", "anova", "mannwhitney", "kruskal", "chi_square"]:
            group_label = "Group Variable"
            value_label = "Dependent Variable"
            group_help = "Select one or more group variables"
            value_help = "Select the dependent variable"
            allow_multiple_group = False
            show_value = True
        elif stat_test == "shapiro":
            group_label = "Variable (not used)"
            value_label = "Dependent Variable"
            group_help = "This field is not used for Shapiro-Wilk test"
            value_help = "Select the variable to test for normality"
            allow_multiple_group = False
            show_value = True
        else:
            group_label = "Independent Variable"
            value_label = "Dependent Variable"
            group_help = "Select one or more independent variables"
            value_help = "Select the dependent variable"
            allow_multiple_group = True
            show_value = True
        
        if allow_multiple_group:
            if isinstance(current_cols['group'], list):
                group_default = [col for col in current_cols['group'] if col in cols]
            elif current_cols['group'] and current_cols['group'] in cols:
                group_default = [current_cols['group']]
            else:
                group_default = []
            
            group_input = ui.input_selectize(
                "modal_stat_group_col",
                group_label,
                choices=cols,
                selected=group_default,
                multiple=True
            )
        else:
            group_default = current_cols['group'] if current_cols['group'] and current_cols['group'] in cols else ""
            group_input = ui.input_select(
                "modal_stat_group_col",
                group_label,
                choices=col_choices,
                selected=group_default
            )
        
        value_default = current_cols['value'] if current_cols['value'] and current_cols['value'] in cols else ""
        
        # Create value input only if needed
        if show_value:
            value_input = ui.div(
                ui.input_select(
                    "modal_stat_value_col",
                    value_label,
                    choices=col_choices,
                    selected=value_default
                ),
                ui.p(value_help, style="font-size: 0.85em; color: #666; margin-top: 5px;"),
            )
        else:
            value_input = ui.p("No dependent variable needed for this test.", 
                             style="font-size: 0.9em; color: #666; font-style: italic;")
        
        m = ui.modal(
            ui.div(
                ui.h5(f"Set Variables for {stat_test}", style="margin-bottom: 15px;"),
                ui.div(
                    group_input,
                    ui.p(group_help, style="font-size: 0.85em; color: #666; margin-top: 5px;"),
                    style="margin-bottom: 20px;"
                ),
                value_input,
            ),
            title="Variable Selection",
            size="m",
            easy_close=True,
            footer=ui.div(
                ui.input_action_button("save_stat_columns", "Save", class_="btn-primary"),
                ui.input_action_button("cancel_stat_columns", "Cancel", class_="btn-secondary"),
                style="display: flex; gap: 10px; justify-content: flex-end;"
            )
        )
        ui.modal_show(m)
    
    @reactive.Effect
    @reactive.event(input.save_stat_columns)
    def save_stat_columns():
        try:
            stat_test = input.stat_test()
            group_col = input.modal_stat_group_col()
            
            # For tests that don't need value column
            if stat_test in ["cronbach", "bartlett", "kmo"]:
                value_col = None
            else:
                value_col = input.modal_stat_value_col()
                value_col = value_col if value_col else None
            
            if isinstance(group_col, (list, tuple)):
                group_col = [col for col in group_col if col]
                if not group_col:
                    group_col = None
            else:
                group_col = group_col if group_col else None
            
            stat_columns.set({'group': group_col, 'value': value_col})
            ui.modal_remove()
            
            if group_col or value_col:
                group_display = ', '.join(group_col) if isinstance(group_col, list) else (group_col or 'None')
                if value_col:
                    ui.notification_show(f"Variables saved: Group={group_display}, Value={value_col}", type="message")
                else:
                    ui.notification_show(f"Variables saved: {group_display}", type="message")
            else:
                ui.notification_show("No variables selected", type="warning")
        except Exception as e:
            ui.notification_show(f"Error saving variables: {str(e)}", type="error")
    
    @reactive.Effect
    @reactive.event(input.cancel_stat_columns)
    def cancel_stat_columns():
        ui.modal_remove()
    
    @reactive.Effect
    @reactive.event(input.run_stat)
    def run_statistical_test():
        df = data.get()
        if df is None:
            ui.notification_show("No data loaded", type="error")
            return
        
        stat_test = input.stat_test()
        cols = stat_columns.get()
        group_col = cols['group']
        value_col = cols['value']
        
        if group_col and not isinstance(group_col, list):
            group_col_list = [group_col]
        else:
            group_col_list = group_col if group_col else []
        
        # Check requirements for each test
        if stat_test in ["cronbach", "bartlett", "kmo"]:
            if len(group_col_list) < 2:
                ui.notification_show(f"{stat_test.upper()} requires at least 2 variables", type="error")
                return
        elif not group_col_list or (not value_col and stat_test not in ["cronbach", "bartlett", "kmo"]):
            ui.notification_show("Please set variables using 'Set Variables' button", type="error")
            return
        elif stat_test == "anova_two" and len(group_col_list) != 2:
            ui.notification_show("Two-Way ANOVA requires exactly 2 group variables", type="error")
            return
        
        ui.notification_show(f"Running {stat_test}...", type="message")
    
    @output
    @render.text
    def stat_results():
        df = data.get()
        if df is None:
            return "No data loaded"
        
        if input.run_stat() == 0:
            return "Click 'Run Test' to perform statistical analysis"
        
        stat_test = input.stat_test()
        cols = stat_columns.get()
        group_col = cols['group']
        value_col = cols['value']
        
        if group_col and not isinstance(group_col, list):
            group_col_list = [group_col]
        else:
            group_col_list = group_col if group_col else []
        
        if not group_col_list or (not value_col and stat_test not in ["cronbach", "bartlett", "kmo"]):
            return "Please set variables using 'Set Variables' button"
        
        # Cronbach's Alpha
        if stat_test == "cronbach":
            if len(group_col_list) < 2:
                return "Cronbach's Alpha requires at least 2 variables."
            
            try:
                selected_data = df[group_col_list].dropna()
                
                # Calculate Cronbach's Alpha
                n_items = len(group_col_list)
                item_variances = selected_data.var(axis=0, ddof=1)
                total_variance = selected_data.sum(axis=1).var(ddof=1)
                
                cronbach_alpha = (n_items / (n_items - 1)) * (1 - item_variances.sum() / total_variance)
                
                result = f"Cronbach's Alpha Results\n"
                result += f"="*50 + "\n"
                result += f"Variables: {', '.join(group_col_list)}\n"
                result += f"Number of items: {n_items}\n"
                result += f"Number of observations: {len(selected_data)}\n"
                result += f"\nCronbach's Alpha: {cronbach_alpha:.4f}\n"
                result += f"\nInterpretation:\n"
                if cronbach_alpha >= 0.9:
                    result += "Excellent internal consistency"
                elif cronbach_alpha >= 0.8:
                    result += "Good internal consistency"
                elif cronbach_alpha >= 0.7:
                    result += "Acceptable internal consistency"
                elif cronbach_alpha >= 0.6:
                    result += "Questionable internal consistency"
                elif cronbach_alpha >= 0.5:
                    result += "Poor internal consistency"
                else:
                    result += "Unacceptable internal consistency"
                
                return result
                
            except Exception as e:
                return f"Error calculating Cronbach's Alpha: {str(e)}"
        
        # Bartlett's Test
        if stat_test == "bartlett":
            if len(group_col_list) < 2:
                return "Bartlett's Test requires at least 2 variables."
            
            try:
                selected_data = df[group_col_list].dropna()
                
                # Calculate correlation matrix
                corr_matrix = selected_data.corr()
                
                # Bartlett's test statistic
                n = len(selected_data)
                p = len(group_col_list)
                
                # Determinant of correlation matrix
                det_corr = np.linalg.det(corr_matrix)
                
                # Chi-square statistic
                chi_square = -((n - 1) - (2 * p + 5) / 6) * np.log(det_corr)
                
                # Degrees of freedom
                df_bart = p * (p - 1) / 2
                
                # P-value
                from scipy.stats import chi2
                p_value = 1 - chi2.cdf(chi_square, df_bart)
                
                result = f"Bartlett's Test of Sphericity Results\n"
                result += f"="*50 + "\n"
                result += f"Variables: {', '.join(group_col_list)}\n"
                result += f"Number of variables: {p}\n"
                result += f"Number of observations: {n}\n"
                result += f"\nChi-square statistic: {chi_square:.4f}\n"
                result += f"Degrees of freedom: {int(df_bart)}\n"
                result += f"P-value: {p_value:.4f}\n"
                result += f"\nInterpretation: "
                if p_value < 0.05:
                    result += "Variables are correlated (p < 0.05)\nSuitable for factor analysis"
                else:
                    result += "Variables are not sufficiently correlated (p >= 0.05)\nNot suitable for factor analysis"
                
                return result
                
            except Exception as e:
                return f"Error calculating Bartlett's Test: {str(e)}"
        
        # KMO Test
        if stat_test == "kmo":
            if len(group_col_list) < 2:
                return "KMO Test requires at least 2 variables."
            
            try:
                selected_data = df[group_col_list].dropna()
                
                # Calculate correlation matrix
                corr_matrix = selected_data.corr().values
                
                # Calculate partial correlation matrix
                corr_inv = np.linalg.inv(corr_matrix)
                partial_corr = np.zeros_like(corr_matrix)
                
                for i in range(len(corr_matrix)):
                    for j in range(len(corr_matrix)):
                        if i != j:
                            partial_corr[i, j] = -corr_inv[i, j] / np.sqrt(corr_inv[i, i] * corr_inv[j, j])
                
                # Calculate KMO for each variable
                kmo_per_variable = []
                for i in range(len(corr_matrix)):
                    sum_sq_corr = np.sum(corr_matrix[i, :] ** 2) - 1  # Exclude diagonal
                    sum_sq_partial = np.sum(partial_corr[i, :] ** 2)
                    kmo_i = sum_sq_corr / (sum_sq_corr + sum_sq_partial)
                    kmo_per_variable.append(kmo_i)
                
                # Overall KMO
                sum_sq_corr_all = np.sum(corr_matrix ** 2) - len(corr_matrix)  # Exclude diagonal
                sum_sq_partial_all = np.sum(partial_corr ** 2)
                kmo_overall = sum_sq_corr_all / (sum_sq_corr_all + sum_sq_partial_all)
                
                result = f"Kaiser-Meyer-Olkin (KMO) Test Results\n"
                result += f"="*50 + "\n"
                result += f"Variables: {', '.join(group_col_list)}\n"
                result += f"Number of variables: {len(group_col_list)}\n"
                result += f"Number of observations: {len(selected_data)}\n"
                result += f"\nOverall KMO: {kmo_overall:.4f}\n"
                result += f"\nKMO by Variable:\n"
                for var, kmo_val in zip(group_col_list, kmo_per_variable):
                    result += f"  {var}: {kmo_val:.4f}\n"
                
                result += f"\nInterpretation of Overall KMO:\n"
                if kmo_overall >= 0.9:
                    result += "Marvelous - Excellent for factor analysis"
                elif kmo_overall >= 0.8:
                    result += "Meritorious - Good for factor analysis"
                elif kmo_overall >= 0.7:
                    result += "Middling - Acceptable for factor analysis"
                elif kmo_overall >= 0.6:
                    result += "Mediocre - Marginal for factor analysis"
                elif kmo_overall >= 0.5:
                    result += "Miserable - Poor for factor analysis"
                else:
                    result += "Unacceptable - Not suitable for factor analysis"
                
                return result
                
            except Exception as e:
                return f"Error calculating KMO: {str(e)}"
        
        # Two-Way ANOVA
        if stat_test == "anova_two":
            if len(group_col_list) != 2:
                return f"Two-Way ANOVA requires exactly 2 group variables. You selected {len(group_col_list)} variable(s)."
            
            factor1 = group_col_list[0]
            factor2 = group_col_list[1]
            
            try:
                anova_data = df[[factor1, factor2, value_col]].dropna()
                
                factor1_levels = anova_data[factor1].unique()
                factor2_levels = anova_data[factor2].unique()
                
                grand_mean = anova_data[value_col].mean()
                n_total = len(anova_data)
                
                # SS_total
                ss_total = ((anova_data[value_col] - grand_mean) ** 2).sum()
                
                # SS_factor1
                ss_factor1 = 0
                for level in factor1_levels:
                    subset = anova_data[anova_data[factor1] == level]
                    ss_factor1 += len(subset) * ((subset[value_col].mean() - grand_mean) ** 2)
                
                # SS_factor2
                ss_factor2 = 0
                for level in factor2_levels:
                    subset = anova_data[anova_data[factor2] == level]
                    ss_factor2 += len(subset) * ((subset[value_col].mean() - grand_mean) ** 2)
                
                # SS_interaction
                ss_interaction = 0
                for lev1 in factor1_levels:
                    for lev2 in factor2_levels:
                        subset = anova_data[(anova_data[factor1] == lev1) & (anova_data[factor2] == lev2)]
                        if len(subset) > 0:
                            f1_mean = anova_data[anova_data[factor1] == lev1][value_col].mean()
                            f2_mean = anova_data[anova_data[factor2] == lev2][value_col].mean()
                            cell_mean = subset[value_col].mean()
                            ss_interaction += len(subset) * ((cell_mean - f1_mean - f2_mean + grand_mean) ** 2)
                
                ss_error = ss_total - ss_factor1 - ss_factor2 - ss_interaction
                
                df_factor1 = len(factor1_levels) - 1
                df_factor2 = len(factor2_levels) - 1
                df_interaction = df_factor1 * df_factor2
                df_error = n_total - (len(factor1_levels) * len(factor2_levels))
                
                ms_factor1 = ss_factor1 / df_factor1 if df_factor1 > 0 else 0
                ms_factor2 = ss_factor2 / df_factor2 if df_factor2 > 0 else 0
                ms_interaction = ss_interaction / df_interaction if df_interaction > 0 else 0
                ms_error = ss_error / df_error if df_error > 0 else 0
                
                f_factor1 = ms_factor1 / ms_error if ms_error > 0 else 0
                f_factor2 = ms_factor2 / ms_error if ms_error > 0 else 0
                f_interaction = ms_interaction / ms_error if ms_error > 0 else 0
                
                from scipy.stats import f as f_dist
                p_factor1 = 1 - f_dist.cdf(f_factor1, df_factor1, df_error) if df_factor1 > 0 and df_error > 0 else 1
                p_factor2 = 1 - f_dist.cdf(f_factor2, df_factor2, df_error) if df_factor2 > 0 and df_error > 0 else 1
                p_interaction = 1 - f_dist.cdf(f_interaction, df_interaction, df_error) if df_interaction > 0 and df_error > 0 else 1
                
                result = f"Two-Way ANOVA Results\n"
                result += f"="*50 + "\n"
                result += f"Factor 1: {factor1}\n"
                result += f"Factor 2: {factor2}\n"
                result += f"Dependent Variable: {value_col}\n"
                result += f"Total observations: {n_total}\n\n"
                
                result += f"Main Effect - {factor1}:\n"
                result += f"  F-statistic: {f_factor1:.4f}\n"
                result += f"  P-value: {p_factor1:.4f}\n"
                result += f"  Interpretation: "
                if p_factor1 < 0.05:
                    result += f"Significant effect (p < 0.05)\n\n"
                else:
                    result += f"No significant effect (p >= 0.05)\n\n"
                
                result += f"Main Effect - {factor2}:\n"
                result += f"  F-statistic: {f_factor2:.4f}\n"
                result += f"  P-value: {p_factor2:.4f}\n"
                result += f"  Interpretation: "
                if p_factor2 < 0.05:
                    result += f"Significant effect (p < 0.05)\n\n"
                else:
                    result += f"No significant effect (p >= 0.05)\n\n"
                
                result += f"Interaction Effect ({factor1} × {factor2}):\n"
                result += f"  F-statistic: {f_interaction:.4f}\n"
                result += f"  P-value: {p_interaction:.4f}\n"
                result += f"  Interpretation: "
                if p_interaction < 0.05:
                    result += f"Significant interaction (p < 0.05)"
                else:
                    result += f"No significant interaction (p >= 0.05)"
                
                return result
                
            except Exception as e:
                return f"Error running Two-Way ANOVA: {str(e)}"
        
        # Other tests
        all_results = []
        
        for idx, group_col in enumerate(group_col_list, 1):
            try:
                result_header = ""
                if len(group_col_list) > 1:
                    result_header = f"\n{'='*60}\n"
                    result_header += f"TEST {idx}/{len(group_col_list)}: Group Variable = '{group_col}'\n"
                    result_header += f"{'='*60}\n"
                
                if stat_test == "ttest":
                    groups = df[group_col].unique()
                    if len(groups) != 2:
                        result = f"T-Test requires exactly 2 groups. Found {len(groups)} groups in '{group_col}'."
                    else:
                        group1 = df[df[group_col] == groups[0]][value_col].dropna()
                        group2 = df[df[group_col] == groups[1]][value_col].dropna()
                        
                        t_stat, p_value = stats.ttest_ind(group1, group2)
                        
                        result = f"Independent T-Test Results\n"
                        result += f"Group Variable: {group_col}\n"
                        result += f"Dependent Variable: {value_col}\n"
                        result += f"-"*50 + "\n"
                        result += f"Group 1: {groups[0]} (n={len(group1)})\n"
                        result += f"Group 2: {groups[1]} (n={len(group2)})\n"
                        result += f"\nT-statistic: {t_stat:.4f}\n"
                        result += f"P-value: {p_value:.4f}\n"
                        result += f"\nInterpretation: "
                        if p_value < 0.05:
                            result += "Significant difference (p < 0.05)"
                        else:
                            result += "No significant difference (p >= 0.05)"
                    
                    all_results.append(result_header + result)
                
                elif stat_test == "ttest_paired":
                    groups = df[group_col].unique()
                    if len(groups) != 2:
                        result = f"Paired T-Test requires exactly 2 groups. Found {len(groups)} groups in '{group_col}'."
                    else:
                        group1 = df[df[group_col] == groups[0]][value_col].dropna()
                        group2 = df[df[group_col] == groups[1]][value_col].dropna()
                        
                        if len(group1) != len(group2):
                            result = "Paired T-Test requires equal sample sizes"
                        else:
                            t_stat, p_value = stats.ttest_rel(group1, group2)
                            
                            result = f"Paired T-Test Results\n"
                            result += f"Group Variable: {group_col}\n"
                            result += f"Dependent Variable: {value_col}\n"
                            result += f"-"*50 + "\n"
                            result += f"Sample size: {len(group1)}\n"
                            result += f"\nT-statistic: {t_stat:.4f}\n"
                            result += f"P-value: {p_value:.4f}\n"
                            result += f"\nInterpretation: "
                            if p_value < 0.05:
                                result += "Significant difference (p < 0.05)"
                            else:
                                result += "No significant difference (p >= 0.05)"
                    
                    all_results.append(result_header + result)
                
                elif stat_test == "anova":
                    groups = [group[value_col].dropna().values for name, group in df.groupby(group_col)]
                    
                    if len(groups) < 2:
                        result = f"ANOVA requires at least 2 groups in '{group_col}'"
                    else:
                        f_stat, p_value = stats.f_oneway(*groups)
                        
                        result = f"One-Way ANOVA Results\n"
                        result += f"Group Variable: {group_col}\n"
                        result += f"Dependent Variable: {value_col}\n"
                        result += f"-"*50 + "\n"
                        result += f"Number of groups: {len(groups)}\n"
                        result += f"Total observations: {sum(len(g) for g in groups)}\n"
                        result += f"\nF-statistic: {f_stat:.4f}\n"
                        result += f"P-value: {p_value:.4f}\n"
                        result += f"\nInterpretation: "
                        if p_value < 0.05:
                            result += "Significant difference between groups (p < 0.05)"
                        else:
                            result += "No significant difference between groups (p >= 0.05)"
                    
                    all_results.append(result_header + result)
                
                elif stat_test == "chi_square":
                    contingency_table = pd.crosstab(df[group_col], df[value_col])
                    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
                    
                    result = f"Chi-Square Test of Independence\n"
                    result += f"Independent Variable: {group_col}\n"
                    result += f"Dependent Variable: {value_col}\n"
                    result += f"-"*50 + "\n"
                    result += f"Chi-square statistic: {chi2:.4f}\n"
                    result += f"P-value: {p_value:.4f}\n"
                    result += f"Degrees of freedom: {dof}\n"
                    result += f"\nInterpretation: "
                    if p_value < 0.05:
                        result += "Variables are dependent (p < 0.05)"
                    else:
                        result += "Variables are independent (p >= 0.05)"
                    
                    all_results.append(result_header + result)
                
                elif stat_test == "mannwhitney":
                    groups = df[group_col].unique()
                    if len(groups) != 2:
                        result = f"Mann-Whitney U Test requires exactly 2 groups. Found {len(groups)} groups in '{group_col}'."
                    else:
                        group1 = df[df[group_col] == groups[0]][value_col].dropna()
                        group2 = df[df[group_col] == groups[1]][value_col].dropna()
                        
                        u_stat, p_value = stats.mannwhitneyu(group1, group2)
                        
                        result = f"Mann-Whitney U Test Results\n"
                        result += f"Group Variable: {group_col}\n"
                        result += f"Dependent Variable: {value_col}\n"
                        result += f"-"*50 + "\n"
                        result += f"Group 1: {groups[0]} (n={len(group1)})\n"
                        result += f"Group 2: {groups[1]} (n={len(group2)})\n"
                        result += f"\nU-statistic: {u_stat:.4f}\n"
                        result += f"P-value: {p_value:.4f}\n"
                        result += f"\nInterpretation: "
                        if p_value < 0.05:
                            result += "Significant difference (p < 0.05)"
                        else:
                            result += "No significant difference (p >= 0.05)"
                    
                    all_results.append(result_header + result)
                
                elif stat_test == "kruskal":
                    groups = [group[value_col].dropna().values for name, group in df.groupby(group_col)]
                    
                    if len(groups) < 2:
                        result = f"Kruskal-Wallis Test requires at least 2 groups in '{group_col}'"
                    else:
                        h_stat, p_value = stats.kruskal(*groups)
                        
                        result = f"Kruskal-Wallis H Test Results\n"
                        result += f"Group Variable: {group_col}\n"
                        result += f"Dependent Variable: {value_col}\n"
                        result += f"-"*50 + "\n"
                        result += f"Number of groups: {len(groups)}\n"
                        result += f"Total observations: {sum(len(g) for g in groups)}\n"
                        result += f"\nH-statistic: {h_stat:.4f}\n"
                        result += f"P-value: {p_value:.4f}\n"
                        result += f"\nInterpretation: "
                        if p_value < 0.05:
                            result += "Significant difference between groups (p < 0.05)"
                        else:
                            result += "No significant difference between groups (p >= 0.05)"
                    
                    all_results.append(result_header + result)
                
                elif stat_test == "shapiro":
                    stat, p_value = stats.shapiro(df[value_col].dropna())
                    
                    result = f"Shapiro-Wilk Normality Test\n"
                    result += f"Dependent Variable: {value_col}\n"
                    result += f"-"*50 + "\n"
                    result += f"Test statistic: {stat:.4f}\n"
                    result += f"P-value: {p_value:.4f}\n"
                    result += f"\nInterpretation: "
                    if p_value < 0.05:
                        result += "Data is NOT normally distributed (p < 0.05)"
                    else:
                        result += "Data is normally distributed (p >= 0.05)"
                    
                    all_results.append(result_header + result)
                    break
                
            except Exception as e:
                error_msg = f"Error running test for '{group_col}': {str(e)}"
                all_results.append(result_header + error_msg)
        
        return "\n\n".join(all_results)
    
    @output
    @render.data_frame
    def group_stats():
        df = data.get()
        if df is None or input.run_stat() == 0:
            return None
        
        cols = stat_columns.get()
        group_col = cols['group']
        value_col = cols['value']
        stat_test = input.stat_test()
        
        if group_col and not isinstance(group_col, list):
            group_col_list = [group_col]
        else:
            group_col_list = group_col if group_col else []
        
        if not group_col_list or (not value_col and stat_test not in ["cronbach", "bartlett", "kmo"]):
            return None
        
        try:
            # For reliability/adequacy tests, show descriptive statistics
            if stat_test in ["cronbach", "bartlett", "kmo"]:
                stats_df = df[group_col_list].describe().T
                stats_df = stats_df.reset_index()
                stats_df.columns = ['Variable', 'Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
                
                for col in ['Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']:
                    if col in stats_df.columns:
                        stats_df[col] = stats_df[col].round(4)
                
                return render.DataGrid(stats_df, width="100%", row_selection_mode="none")
            
            if stat_test == "shapiro":
                stats_df = pd.DataFrame({
                    'Statistic': ['Count', 'Mean', 'Std', 'Min', 'Max'],
                    'Value': [
                        df[value_col].count(),
                        df[value_col].mean(),
                        df[value_col].std(),
                        df[value_col].min(),
                        df[value_col].max()
                    ]
                })
                for col in ['Value']:
                    if col in stats_df.columns:
                        stats_df[col] = stats_df[col].round(4)
                return render.DataGrid(stats_df, width="100%", row_selection_mode="none")
            
            if stat_test == "anova_two" and len(group_col_list) == 2:
                factor1 = group_col_list[0]
                factor2 = group_col_list[1]
                
                interaction_stats = df.groupby([factor1, factor2])[value_col].agg([
                    ('Count', 'count'),
                    ('Mean', 'mean'),
                    ('Std', 'std'),
                    ('Min', 'min'),
                    ('Max', 'max')
                ]).reset_index()
                
                for col in ['Mean', 'Std', 'Min', 'Max']:
                    if col in interaction_stats.columns:
                        interaction_stats[col] = interaction_stats[col].round(4)
                
                return render.DataGrid(interaction_stats, width="100%", row_selection_mode="none")
            
            all_stats = []
            for group_col in group_col_list:
                group_stats_df = df.groupby(group_col)[value_col].agg([
                    ('Count', 'count'),
                    ('Mean', 'mean'),
                    ('Std', 'std'),
                    ('Min', 'min'),
                    ('Max', 'max')
                ]).reset_index()
                
                group_stats_df.insert(0, 'Group_Variable', group_col)
                
                for col in ['Mean', 'Std', 'Min', 'Max']:
                    if col in group_stats_df.columns:
                        group_stats_df[col] = group_stats_df[col].round(4)
                
                all_stats.append(group_stats_df)
            
            if all_stats:
                combined_stats = pd.concat(all_stats, ignore_index=True)
                return render.DataGrid(combined_stats, width="100%", row_selection_mode="none")
            
            return None
        except Exception as e:
            return None
    @reactive.Effect
    @reactive.event(input.set_dim_settings)
    def show_dim_settings_modal():
        df = data.get()
        if df is None:
            ui.notification_show("Please load data first", type="error")
            return
        
        dim_method = input.dim_method()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            ui.notification_show("Need at least 2 numeric variables", type="error")
            return
        
        current_settings = dim_settings.get()
        
        m = ui.modal(
            ui.div(
                ui.h5(f"Settings for {dim_method.upper()}", style="margin-bottom: 15px;"),
                ui.div(
                    ui.input_selectize(
                        "dim_variables",
                        "Select Variables",
                        choices=numeric_cols,
                        selected=current_settings.get('variables', []),
                        multiple=True
                    ),
                    ui.p("Select numeric variables to include in the analysis", 
                         style="font-size: 0.85em; color: #666; margin-top: 5px;"),
                    style="margin-bottom: 15px;"
                ),
                ui.div(
                    ui.input_numeric(
                        "dim_n_components",
                        "Number of Components",
                        value=current_settings.get('n_components', 2),
                        min=1,
                        max=min(len(numeric_cols), 10)
                    ),
                    ui.p(f"Maximum: {min(len(numeric_cols), 10)}", 
                         style="font-size: 0.85em; color: #666; margin-top: 5px;"),
                    style="margin-bottom: 15px;"
                ),
                ui.div(
                    ui.input_select(
                        "dim_rotation",
                        "Rotation Method (FA only)",
                        choices={
                            "varimax": "Varimax",
                            "promax": "Promax",
                            "quartimax": "Quartimax"
                        },
                        selected=current_settings.get('rotation', 'varimax')
                    ),
                    ui.p("Rotation method for Factor Analysis", 
                         style="font-size: 0.85em; color: #666; margin-top: 5px;"),
                ) if dim_method == "fa" else None,
            ),
            title="Dimensionality Reduction Settings",
            size="m",
            easy_close=True,
            footer=ui.div(
                ui.input_action_button("save_dim_settings", "Save", class_="btn-primary"),
                ui.input_action_button("cancel_dim_settings", "Cancel", class_="btn-secondary"),
                style="display: flex; gap: 10px; justify-content: flex-end;"
            )
        )
        ui.modal_show(m)
    
    @reactive.Effect
    @reactive.event(input.save_dim_settings)
    def save_dim_settings():
        try:
            variables = input.dim_variables()
            n_components = input.dim_n_components()
            rotation = input.dim_rotation() if input.dim_method() == "fa" else "varimax"
            
            if not variables or len(variables) < 2:
                ui.notification_show("Please select at least 2 variables", type="error")
                return
            
            if n_components > len(variables):
                ui.notification_show(f"Number of components cannot exceed number of variables ({len(variables)})", type="error")
                return
            
            dim_settings.set({
                'n_components': n_components,
                'rotation': rotation,
                'variables': list(variables)
            })
            
            ui.modal_remove()
            ui.notification_show(f"Settings saved: {len(variables)} variables, {n_components} components", type="message")
        except Exception as e:
            ui.notification_show(f"Error saving settings: {str(e)}", type="error")
    
    @reactive.Effect
    @reactive.event(input.cancel_dim_settings)
    def cancel_dim_settings():
        ui.modal_remove()
    
    @reactive.Effect
    @reactive.event(input.show_dim_info)
    def show_dim_info_modal():
        m = ui.modal(
            ui.div(
                ui.h5("Dimensionality Reduction Methods", style="font-weight: bold; margin-bottom: 15px; color: #2c3e50;"),
                
                ui.p("• PCA: Linear dimensionality reduction that finds directions of maximum variance", 
                     style="font-size: 0.95em; margin-bottom: 10px; margin-left: 10px;"),
                ui.p("• FA: Identifies underlying latent factors that explain patterns of correlations", 
                     style="font-size: 0.95em; margin-bottom: 0; margin-left: 10px;"),
            ),
            title="Dimensionality Reduction Information",
            size="m",
            easy_close=True,
            footer=ui.div(
                ui.input_action_button("close_dim_info", "Close", class_="btn-primary"),
                style="display: flex; justify-content: flex-end;"
            )
        )
        ui.modal_show(m)
    
    @reactive.Effect
    @reactive.event(input.close_dim_info)
    def close_dim_info():
        ui.modal_remove()
    
    @reactive.Effect
    @reactive.event(input.run_dim_reduction)
    def run_dimensionality_reduction():
        df = data.get()
        if df is None:
            ui.notification_show("No data loaded", type="error")
            return
        
        settings = dim_settings.get()
        variables = settings.get('variables')
        
        if not variables or len(variables) < 2:
            ui.notification_show("Please set variables first", type="error")
            return
        
        n_components = settings.get('n_components', 2)
        method = input.dim_method()
        
        try:
            # Select only the specified variables
            X = df[variables].dropna()
            
            if len(X) == 0:
                ui.notification_show("No data after removing missing values", type="error")
                return
            
            # Standardize the data
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            if method == "pca":
                from sklearn.decomposition import PCA
                
                model = PCA(n_components=n_components)
                components = model.fit_transform(X_scaled)
                
                # Create loadings dataframe
                loadings_df = pd.DataFrame(
                    model.components_.T,
                    columns=[f'PC{i+1}' for i in range(n_components)],
                    index=variables
                )
                
                # Explained variance
                variance_df = pd.DataFrame({
                    'Component': [f'PC{i+1}' for i in range(n_components)],
                    'Eigenvalue': model.explained_variance_,
                    'Variance %': model.explained_variance_ratio_ * 100,
                    'Cumulative %': np.cumsum(model.explained_variance_ratio_) * 100
                })
                
                # Create new columns in the original dataframe
                df_copy = data.get().copy()
                for i in range(n_components):
                    col_name = f'PC{i+1}'
                    # Initialize with NaN
                    df_copy[col_name] = np.nan
                    # Fill only the rows that had complete data
                    df_copy.loc[df[variables].dropna().index, col_name] = components[:, i]
                
                data.set(df_copy)
                
                # Update column choices
                cols = df_copy.columns.tolist()
                ui.update_select("delete_col", choices=cols)
                ui.update_select("missing_col", choices=cols)
                ui.update_select("outlier_col", choices=cols)
                ui.update_select("encode_col", choices=cols)
                
                dim_results.set({
                    'method': 'PCA',
                    'loadings': loadings_df,
                    'variance': variance_df,
                    'eigenvalues': model.explained_variance_,
                    'n_components': n_components
                })
                
                ui.notification_show(f"PCA completed! {n_components} components added to data", type="message")
                
            else:  # FA
                from sklearn.decomposition import FactorAnalysis
                
                model = FactorAnalysis(n_components=n_components, rotation=settings.get('rotation', 'varimax'))
                factors = model.fit_transform(X_scaled)
                
                # Create loadings dataframe
                loadings_df = pd.DataFrame(
                    model.components_.T,
                    columns=[f'Factor{i+1}' for i in range(n_components)],
                    index=variables
                )
                
                # Calculate explained variance for FA
                total_variance = np.sum(np.var(X_scaled, axis=0))
                factor_variance = np.var(factors, axis=0)
                variance_ratio = factor_variance / total_variance
                
                variance_df = pd.DataFrame({
                    'Factor': [f'Factor{i+1}' for i in range(n_components)],
                    'Variance': factor_variance,
                    'Variance %': variance_ratio * 100,
                    'Cumulative %': np.cumsum(variance_ratio) * 100
                })
                
                # Create new columns in the original dataframe
                df_copy = data.get().copy()
                for i in range(n_components):
                    col_name = f'Factor{i+1}'
                    df_copy[col_name] = np.nan
                    df_copy.loc[df[variables].dropna().index, col_name] = factors[:, i]
                
                data.set(df_copy)
                
                # Update column choices
                cols = df_copy.columns.tolist()
                ui.update_select("delete_col", choices=cols)
                ui.update_select("missing_col", choices=cols)
                ui.update_select("outlier_col", choices=cols)
                ui.update_select("encode_col", choices=cols)
                
                dim_results.set({
                    'method': 'FA',
                    'loadings': loadings_df,
                    'variance': variance_df,
                    'eigenvalues': factor_variance,
                    'n_components': n_components
                })
                
                ui.notification_show(f"FA completed! {n_components} factors added to data", type="message")
                
        except Exception as e:
            ui.notification_show(f"Error in dimensionality reduction: {str(e)}", type="error")
    
    @output
    @render.plot
    def dim_scree_plot():
        results = dim_results.get()
        if results is None:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        eigenvalues = results['eigenvalues']
        n_components = len(eigenvalues)
        
        x = np.arange(1, n_components + 1)
        ax.plot(x, eigenvalues, 'bo-', linewidth=2, markersize=8)
        ax.axhline(y=1, color='r', linestyle='--', label='Eigenvalue = 1')
        
        ax.set_xlabel('Component Number', fontsize=12)
        ax.set_ylabel('Eigenvalue', fontsize=12)
        ax.set_title(f'Scree Plot ({results["method"]})', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @output
    @render.data_frame
    def dim_loadings():
        results = dim_results.get()
        if results is None:
            return None
        
        loadings_df = results['loadings'].reset_index()
        loadings_df.columns = ['Variable'] + list(loadings_df.columns[1:])
        
        # Round to 3 decimal places
        for col in loadings_df.columns[1:]:
            loadings_df[col] = loadings_df[col].round(3)
        
        return render.DataGrid(loadings_df, width="100%", row_selection_mode="none")
    
    @output
    @render.data_frame
    def dim_variance():
        results = dim_results.get()
        if results is None:
            return None
        
        variance_df = results['variance'].copy()
        
        # Round numeric columns
        for col in ['Eigenvalue', 'Variance', 'Variance %', 'Cumulative %']:
            if col in variance_df.columns:
                variance_df[col] = variance_df[col].round(3)
        
        return render.DataGrid(variance_df, width="100%", row_selection_mode="none")    
    @reactive.Effect
    @reactive.event(input.show_model_info)
    def show_model_info_modal():
        m = ui.modal(
            ui.div(
                ui.h5("Classification Models", style="font-weight: bold; margin-bottom: 15px; color: #2c3e50;"),
                ui.p("• Logistic Regression: Linear model for binary/multi-class classification", style="font-size: 0.9em; margin-bottom: 8px; margin-left: 10px;"),
                ui.p("• Support Vector Machine: Finds optimal hyperplane for classification", style="font-size: 0.9em; margin-bottom: 8px; margin-left: 10px;"),
                ui.p("• Decision Tree: Tree-based model using feature splits", style="font-size: 0.9em; margin-bottom: 8px; margin-left: 10px;"),
                ui.p("• Random Forest: Ensemble of decision trees", style="font-size: 0.9em; margin-bottom: 8px; margin-left: 10px;"),
                ui.p("• Stochastic Gradient Descent: Gradient-based classifier", style="font-size: 0.9em; margin-bottom: 8px; margin-left: 10px;"),
                ui.p("• Gradient Boosting: Sequential ensemble method", style="font-size: 0.9em; margin-bottom: 8px; margin-left: 10px;"),
                ui.p("• XGBoost: Optimized gradient boosting library", style="font-size: 0.9em; margin-bottom: 8px; margin-left: 10px;"),
                ui.p("• Gaussian Process: Probabilistic model using kernels", style="font-size: 0.9em; margin-bottom: 8px; margin-left: 10px;"),
                ui.p("• LightGBM: Fast gradient boosting framework", style="font-size: 0.9em; margin-bottom: 20px; margin-left: 10px;"),
                ui.hr(),
                ui.h5("Regression Models", style="font-weight: bold; margin-bottom: 15px; margin-top: 15px; color: #2c3e50;"),
                ui.p("• Linear Regression: Basic linear relationship model", style="font-size: 0.9em; margin-bottom: 8px; margin-left: 10px;"),
                ui.p("• Ridge: Linear regression with L2 regularization", style="font-size: 0.9em; margin-bottom: 8px; margin-left: 10px;"),
                ui.p("• Lasso: Linear regression with L1 regularization", style="font-size: 0.9em; margin-bottom: 8px; margin-left: 10px;"),
                ui.p("• Support Vector Regression: Support vector method for regression", style="font-size: 0.9em; margin-bottom: 8px; margin-left: 10px;"),
                ui.p("• Decision Tree: Tree-based regression model", style="font-size: 0.9em; margin-bottom: 8px; margin-left: 10px;"),
                ui.p("• Random Forest: Ensemble of regression trees", style="font-size: 0.9em; margin-bottom: 8px; margin-left: 10px;"),
                ui.p("• Stochastic Gradient Descent: Gradient-based regressor", style="font-size: 0.9em; margin-bottom: 8px; margin-left: 10px;"),
                ui.p("• Gradient Boosting: Sequential ensemble for regression", style="font-size: 0.9em; margin-bottom: 8px; margin-left: 10px;"),
                ui.p("• XGBoost: Optimized gradient boosting for regression", style="font-size: 0.9em; margin-bottom: 8px; margin-left: 10px;"),
                ui.p("• Gaussian Process: Probabilistic regression model", style="font-size: 0.9em; margin-bottom: 8px; margin-left: 10px;"),
                ui.p("• LightGBM: Fast gradient boosting for regression", style="font-size: 0.9em; margin-bottom: 0; margin-left: 10px;"),
                style="padding: 20px; max-height: 600px; overflow-y: auto;"
            ),
            title="Model Information",
            size="l",
            easy_close=True,
            footer=ui.div(
                ui.input_action_button("close_model_info", "Close", class_="btn-secondary"),
                style="display: flex; justify-content: flex-end;"
            )
        )
        ui.modal_show(m)
    
    @reactive.Effect
    @reactive.event(input.close_model_info)
    def close_model_info_modal():
        ui.modal_remove()
    
    @reactive.Effect
    @reactive.event(input.set_params)
    def show_params_modal():
        model_name = input.model_name()
        target_type = input.target_type()
        
        try:
            if target_type == "categorical":
                if model_name == "logistic regression":
                    model = LogisticRegression()
                elif model_name == "support vector machine":
                    model = SVC()
                elif model_name == "decision tree":
                    model = DecisionTreeClassifier()
                elif model_name == "random forest":
                    model = RandomForestClassifier()
                elif model_name == "stochastic gradient descent":
                    model = SGDClassifier()
                elif model_name == "gradient boosting":
                    model = GradientBoostingClassifier()
                elif model_name == "xgboost":
                    if not XGBOOST_AVAILABLE:
                        ui.notification_show("XGBoost not installed", type="error")
                        return
                    model = XGBClassifier()
                elif model_name == "gaussian process":
                    model = GaussianProcessClassifier()
                elif model_name == "lgbm":
                    if not LGBM_AVAILABLE:
                        ui.notification_show("LightGBM not installed", type="error")
                        return
                    model = LGBMClassifier()
                else:
                    ui.notification_show("Model not compatible with classification", type="error")
                    return
            else:
                if model_name == "linear regression":
                    model = LinearRegression()
                elif model_name == "ridge":
                    model = Ridge()
                elif model_name == "lasso":
                    model = Lasso()
                elif model_name == "support vector machine":
                    model = SVR()
                elif model_name == "decision tree":
                    model = DecisionTreeRegressor()
                elif model_name == "random forest":
                    model = RandomForestRegressor()
                elif model_name == "stochastic gradient descent":
                    model = SGDRegressor()
                elif model_name == "gradient boosting":
                    model = GradientBoostingRegressor()
                elif model_name == "xgboost":
                    if not XGBOOST_AVAILABLE:
                        ui.notification_show("XGBoost not installed", type="error")
                        return
                    model = XGBRegressor()
                elif model_name == "gaussian process":
                    model = GaussianProcessRegressor()
                elif model_name == "lgbm":
                    if not LGBM_AVAILABLE:
                        ui.notification_show("LightGBM not installed", type="error")
                        return
                    model = LGBMRegressor()
                else:
                    ui.notification_show("Model not compatible with regression", type="error")
                    return
            
            params = model.get_params()
            
            param_constraints = {
                'penalty': ['l1', 'l2', 'elasticnet', 'None'],
                'dual': ['True', 'False'],
                'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
                'multi_class': ['auto', 'ovr', 'multinomial'],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
                'criterion': ['gini', 'entropy', 'log_loss', 'squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                'splitter': ['best', 'random'],
                'max_features': ['auto', 'sqrt', 'log2', 'None'],
                'loss': ['hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron', 
                        'squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
                'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
                'fit_intercept': ['True', 'False'],
                'normalize': ['True', 'False'],
                'copy_X': ['True', 'False'],
                'positive': ['True', 'False'],
                'oob_score': ['True', 'False'],
                'bootstrap': ['True', 'False'],
                'warm_start': ['True', 'False'],
                'early_stopping': ['True', 'False'],
                'shuffle': ['True', 'False'],
                'average': ['True', 'False'],
            }
            
            param_inputs = []
            for param_name, param_value in params.items():
                if param_value is None:
                    default_val = "None"
                elif isinstance(param_value, bool):
                    default_val = str(param_value)
                elif isinstance(param_value, (int, float)):
                    default_val = str(param_value)
                else:
                    default_val = str(param_value)
                
                if param_name in param_constraints and param_constraints[param_name]:
                    choices = param_constraints[param_name]
                    if default_val not in choices:
                        choices.append(default_val)
                    
                    param_inputs.append(
                        ui.div(
                            ui.input_select(
                                f"param_{param_name}",
                                param_name,
                                choices={choice: choice for choice in choices},
                                selected=default_val
                            ),
                            style="margin-bottom: 10px;"
                        )
                    )
                elif isinstance(param_value, bool) or param_name in ['dual', 'fit_intercept', 'normalize']:
                    param_inputs.append(
                        ui.div(
                            ui.input_select(
                                f"param_{param_name}",
                                param_name,
                                choices={'True': 'True', 'False': 'False'},
                                selected=default_val
                            ),
                            style="margin-bottom: 10px;"
                        )
                    )
                else:
                    param_inputs.append(
                        ui.div(
                            ui.input_text(
                                f"param_{param_name}",
                                param_name,
                                value=default_val
                            ),
                            style="margin-bottom: 10px;"
                        )
                    )
            
            m = ui.modal(
                ui.div(
                    ui.h4(f"Parameters for {model_name.title()}", style="margin-bottom: 15px;"),
                    ui.p("Edit the parameters below. Use 'None' for None values.", 
                         style="font-size: 0.9em; color: #666;"),
                    ui.div(
                        *param_inputs,
                        style="max-height: 500px; overflow-y: auto; padding: 10px;"
                    )
                ),
                title="Model Parameters",
                size="l",
                easy_close=True,
                footer=ui.div(
                    ui.input_action_button("save_params", "Save Parameters", class_="btn-primary"),
                    ui.input_action_button("cancel_params", "Cancel", class_="btn-secondary"),
                    style="display: flex; gap: 10px; justify-content: flex-end;"
                )
            )
            ui.modal_show(m)
            
        except Exception as e:
            ui.notification_show(f"Error loading parameters: {str(e)}", type="error")
    
    @reactive.Effect
    @reactive.event(input.save_params)
    def save_parameters():
        model_name = input.model_name()
        target_type = input.target_type()
        
        try:
            if target_type == "categorical":
                if model_name == "logistic regression":
                    model = LogisticRegression()
                elif model_name == "support vector machine":
                    model = SVC()
                elif model_name == "decision tree":
                    model = DecisionTreeClassifier()
                elif model_name == "random forest":
                    model = RandomForestClassifier()
                elif model_name == "stochastic gradient descent":
                    model = SGDClassifier()
                elif model_name == "gradient boosting":
                    model = GradientBoostingClassifier()
                elif model_name == "xgboost":
                    model = XGBClassifier()
                elif model_name == "gaussian process":
                    model = GaussianProcessClassifier()
                elif model_name == "lgbm":
                    model = LGBMClassifier()
                else:
                    return
            else:
                if model_name == "linear regression":
                    model = LinearRegression()
                elif model_name == "ridge":
                    model = Ridge()
                elif model_name == "lasso":
                    model = Lasso()
                elif model_name == "support vector machine":
                    model = SVR()
                elif model_name == "decision tree":
                    model = DecisionTreeRegressor()
                elif model_name == "random forest":
                    model = RandomForestRegressor()
                elif model_name == "stochastic gradient descent":
                    model = SGDRegressor()
                elif model_name == "gradient boosting":
                    model = GradientBoostingRegressor()
                elif model_name == "xgboost":
                    model = XGBRegressor()
                elif model_name == "gaussian process":
                    model = GaussianProcessRegressor()
                elif model_name == "lgbm":
                    model = LGBMRegressor()
                else:
                    return
            
            params = model.get_params()
            modified_params = {}
            
            for param_name in params.keys():
                input_id = f"param_{param_name}"
                try:
                    new_value = input[input_id]()
                    
                    if new_value == "None":
                        modified_params[param_name] = None
                    elif new_value == "True":
                        modified_params[param_name] = True
                    elif new_value == "False":
                        modified_params[param_name] = False
                    else:
                        try:
                            if '.' in new_value:
                                modified_params[param_name] = float(new_value)
                            else:
                                modified_params[param_name] = int(new_value)
                        except ValueError:
                            modified_params[param_name] = new_value
                except:
                    pass
            
            model_params.set(modified_params)
            ui.modal_remove()
            ui.notification_show(f"Parameters saved for {model_name}", type="message")
            
        except Exception as e:
            ui.notification_show(f"Error saving parameters: {str(e)}", type="error")
    
    @reactive.Effect
    @reactive.event(input.cancel_params)
    def cancel_parameters():
        ui.modal_remove()
    
    @reactive.Effect
    @reactive.event(input.set_model_variables)
    def show_model_variables_modal():
        df = data.get()
        if df is None:
            ui.notification_show("Please load data first", type="error")
            return
        
        target_col = input.target_col()
        if not target_col:
            ui.notification_show("Please select target column first", type="error")
            return
        
        # 타겟 변수를 제외한 모든 컬럼
        available_cols = [col for col in df.columns if col != target_col]
        
        if len(available_cols) == 0:
            ui.notification_show("No available variables to select", type="error")
            return
        
        current_vars = model_variables.get()
        
        m = ui.modal(
            ui.div(
                ui.h5("Select Independent Variables", style="margin-bottom: 15px;"),
                ui.input_selectize(
                    "selected_model_vars",
                    "Independent Variables",
                    choices=available_cols,
                    selected=current_vars if current_vars else available_cols,  # 기본값: 모든 변수 선택
                    multiple=True
                ),
                ui.p(f"Target variable: {target_col}", 
                     style="font-size: 0.85em; color: #666; margin-top: 10px;"),
                ui.p("Select variables to use as predictors in the model", 
                     style="font-size: 0.85em; color: #666; margin-top: 5px;"),
            ),
            title="Model Variables Settings",
            size="m",
            easy_close=True,
            footer=ui.div(
                ui.input_action_button("save_model_variables", "Save", class_="btn-primary"),
                ui.input_action_button("cancel_model_variables", "Cancel", class_="btn-secondary"),
                style="display: flex; gap: 10px; justify-content: flex-end;"
            )
        )
        ui.modal_show(m)
    
    @reactive.Effect
    @reactive.event(input.save_model_variables)
    def save_model_variables():
        try:
            selected_vars = input.selected_model_vars()
            
            if not selected_vars or len(selected_vars) == 0:
                ui.notification_show("Please select at least one variable", type="error")
                return
            
            model_variables.set(list(selected_vars))
            ui.modal_remove()
            ui.notification_show(f"Selected {len(selected_vars)} variables", type="message")
        except Exception as e:
            ui.notification_show(f"Error saving variables: {str(e)}", type="error")
    
    @reactive.Effect
    @reactive.event(input.cancel_model_variables)
    def cancel_model_variables():
        ui.modal_remove()
    
    @reactive.Effect
    @reactive.event(input.auto_optimize)
    def auto_optimize_model():
        if not BAYESOPT_AVAILABLE:
            ui.notification_show("bayesian-optimization package not installed. Install: pip install bayesian-optimization", type="error")
            return
        
        df = data.get()
        if df is None:
            ui.notification_show("No data loaded", type="error")
            return
        
        target_col = input.target_col()
        if not target_col:
            ui.notification_show("Please select target column", type="error")
            return
        
        df = df.copy()
        
        # 선택된 독립변수들 가져오기
        selected_vars = model_variables.get()
        if selected_vars:
            X = df[selected_vars]
        else:
            X = df.drop(columns=[target_col])
        
        y = df[target_col]
        
        # Encode categorical variables
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        
        try:
            ui.notification_show("Optimizing parameters... This may take a while.", type="message", duration=3)
            
            model_name = input.model_name()
            target_type = input.target_type()
            
            # Define parameter spaces for each model
            param_bounds = {}
            
            if target_type == "categorical":
                if model_name == "logistic regression":
                    param_bounds = {'C': (0.01, 100)}
                elif model_name == "support vector machine":
                    param_bounds = {'C': (0.1, 100), 'gamma_log': (-4, 1)}
                elif model_name == "decision tree":
                    param_bounds = {'max_depth': (3, 20), 'min_samples_split': (2, 20)}
                elif model_name == "random forest":
                    param_bounds = {'n_estimators': (50, 300), 'max_depth': (3, 20), 'min_samples_split': (2, 20)}
                elif model_name == "gradient boosting":
                    param_bounds = {'n_estimators': (50, 300), 'max_depth': (3, 10), 'learning_rate': (0.01, 0.3)}
                elif model_name == "xgboost":
                    if not XGBOOST_AVAILABLE:
                        ui.notification_show("XGBoost not installed", type="error")
                        return
                    param_bounds = {'n_estimators': (50, 300), 'max_depth': (3, 10), 'learning_rate': (0.01, 0.3)}
                elif model_name == "lgbm":
                    if not LGBM_AVAILABLE:
                        ui.notification_show("LightGBM not installed", type="error")
                        return
                    param_bounds = {'n_estimators': (50, 300), 'max_depth': (3, 10), 'learning_rate': (0.01, 0.3)}
                else:
                    ui.notification_show(f"Auto optimization not available for {model_name}", type="warning")
                    return
            else:  # Regression
                if model_name == "ridge":
                    param_bounds = {'alpha': (0.01, 100)}
                elif model_name == "lasso":
                    param_bounds = {'alpha': (0.01, 100)}
                elif model_name == "support vector machine":
                    param_bounds = {'C': (0.1, 100), 'gamma_log': (-4, 1)}
                elif model_name == "decision tree":
                    param_bounds = {'max_depth': (3, 20), 'min_samples_split': (2, 20)}
                elif model_name == "random forest":
                    param_bounds = {'n_estimators': (50, 300), 'max_depth': (3, 20), 'min_samples_split': (2, 20)}
                elif model_name == "gradient boosting":
                    param_bounds = {'n_estimators': (50, 300), 'max_depth': (3, 10), 'learning_rate': (0.01, 0.3)}
                elif model_name == "xgboost":
                    if not XGBOOST_AVAILABLE:
                        ui.notification_show("XGBoost not installed", type="error")
                        return
                    param_bounds = {'n_estimators': (50, 300), 'max_depth': (3, 10), 'learning_rate': (0.01, 0.3)}
                elif model_name == "lgbm":
                    if not LGBM_AVAILABLE:
                        ui.notification_show("LightGBM not installed", type="error")
                        return
                    param_bounds = {'n_estimators': (50, 300), 'max_depth': (3, 10), 'learning_rate': (0.01, 0.3)}
                else:
                    ui.notification_show(f"Auto optimization not available for {model_name}", type="warning")
                    return
            
            if not param_bounds:
                ui.notification_show("No parameters to optimize", type="warning")
                return
            
            # Define optimization function
            def model_evaluate(**params):
                # Convert parameters to appropriate types
                converted_params = {}
                for key, value in params.items():
                    if key in ['n_estimators', 'max_depth', 'min_samples_split']:
                        converted_params[key] = int(value)
                    elif key == 'gamma_log':
                        converted_params['gamma'] = 10 ** value
                    else:
                        converted_params[key] = value
                
                # Create model with parameters
                if target_type == "categorical":
                    if model_name == "logistic regression":
                        model = LogisticRegression(max_iter=1000, **converted_params)
                    elif model_name == "support vector machine":
                        model = SVC(**converted_params)
                    elif model_name == "decision tree":
                        model = DecisionTreeClassifier(**converted_params)
                    elif model_name == "random forest":
                        model = RandomForestClassifier(**converted_params)
                    elif model_name == "gradient boosting":
                        model = GradientBoostingClassifier(**converted_params)
                    elif model_name == "xgboost":
                        model = XGBClassifier(**converted_params, verbosity=0)
                    elif model_name == "lgbm":
                        model = LGBMClassifier(**converted_params, verbose=-1)
                else:
                    if model_name == "ridge":
                        model = Ridge(**converted_params)
                    elif model_name == "lasso":
                        model = Lasso(**converted_params)
                    elif model_name == "support vector machine":
                        model = SVR(**converted_params)
                    elif model_name == "decision tree":
                        model = DecisionTreeRegressor(**converted_params)
                    elif model_name == "random forest":
                        model = RandomForestRegressor(**converted_params)
                    elif model_name == "gradient boosting":
                        model = GradientBoostingRegressor(**converted_params)
                    elif model_name == "xgboost":
                        model = XGBRegressor(**converted_params, verbosity=0)
                    elif model_name == "lgbm":
                        model = LGBMRegressor(**converted_params, verbose=-1)
                
                # Use cross-validation to evaluate
                if target_type == "categorical":
                    scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
                else:
                    scores = cross_val_score(model, X, y, cv=3, scoring='r2')
                
                return scores.mean()
            
            # Run Bayesian Optimization
            optimizer = BayesianOptimization(
                f=model_evaluate,
                pbounds=param_bounds,
                random_state=input.random_state(),
                verbose=0
            )
            
            optimizer.maximize(init_points=5, n_iter=15)
            
            # Get best parameters
            best_params = optimizer.max['params']
            best_score = optimizer.max['target']
            
            # Convert parameters to display format
            display_params = {}
            for key, value in best_params.items():
                if key in ['n_estimators', 'max_depth', 'min_samples_split']:
                    display_params[key] = int(value)
                elif key == 'gamma_log':
                    display_params['gamma'] = f"{10 ** value:.6f}"
                else:
                    display_params[key] = f"{value:.4f}"
            
            # Show results in modal
            params_text = "\n".join([f"{k}: {v}" for k, v in display_params.items()])
            
            m = ui.modal(
                ui.div(
                    ui.h5("Optimization Results", style="font-weight: bold; margin-bottom: 15px; color: #2c3e50;"),
                    ui.p(f"Model: {model_name.title()}", style="font-size: 1em; margin-bottom: 10px; font-weight: 500;"),
                    ui.p(f"Best CV Score: {best_score:.4f}", style="font-size: 1em; margin-bottom: 15px; font-weight: 500; color: #27ae60;"),
                    ui.h6("Optimal Parameters:", style="font-weight: bold; margin-bottom: 10px; color: #34495e;"),
                    ui.tags.pre(params_text, style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; font-family: monospace; font-size: 0.9em;"),
                    ui.p("These parameters have been saved. Click 'Train Model' to use them.", 
                         style="font-size: 0.85em; color: #666; margin-top: 15px; font-style: italic;"),
                ),
                title="Auto Optimization Complete",
                size="m",
                easy_close=True,
                footer=ui.div(
                    ui.input_action_button("close_optimize_result", "Close", class_="btn-primary"),
                    style="display: flex; justify-content: flex-end;"
                )
            )
            ui.modal_show(m)
            
            # Save optimized parameters
            save_params = {}
            for key, value in best_params.items():
                if key in ['n_estimators', 'max_depth', 'min_samples_split']:
                    save_params[key] = int(value)
                elif key == 'gamma_log':
                    save_params['gamma'] = 10 ** value
                else:
                    save_params[key] = value
            
            model_params.set(save_params)
            
        except Exception as e:
            ui.notification_show(f"Error during optimization: {str(e)}", type="error")
    
    @reactive.Effect
    @reactive.event(input.close_optimize_result)
    def close_optimize_result():
        ui.modal_remove()
    
    @reactive.Effect
    @reactive.event(input.train_model)
    def train_model():
        df = data.get()
        if df is None:
            ui.notification_show("No data loaded", type="error")
            return
        
        target_col = input.target_col()
        if not target_col:
            ui.notification_show("Please select target column", type="error")
            return
        
        df = df.copy()
        
        # 선택된 독립변수들 가져오기
        selected_vars = model_variables.get()
        if selected_vars:
            # 선택된 변수들만 사용
            X = df[selected_vars]
        else:
            # 선택된 변수가 없으면 타겟을 제외한 모든 변수 사용
            X = df.drop(columns=[target_col])
        
        y = df[target_col]
        
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=input.test_size(),
                random_state=input.random_state()
            )
            
            model_name = input.model_name()
            target_type = input.target_type()
            
            saved_params = model_params.get()
            
            if target_type == "categorical":
                if model_name == "logistic regression":
                    model = LogisticRegression(max_iter=1000)
                elif model_name == "support vector machine":
                    model = SVC()
                elif model_name == "decision tree":
                    model = DecisionTreeClassifier()
                elif model_name == "random forest":
                    model = RandomForestClassifier()
                elif model_name == "stochastic gradient descent":
                    model = SGDClassifier(max_iter=1000)
                elif model_name == "gradient boosting":
                    model = GradientBoostingClassifier()
                elif model_name == "xgboost":
                    if not XGBOOST_AVAILABLE:
                        ui.notification_show("XGBoost not installed", type="error")
                        return
                    model = XGBClassifier()
                elif model_name == "gaussian process":
                    model = GaussianProcessClassifier()
                elif model_name == "lgbm":
                    if not LGBM_AVAILABLE:
                        ui.notification_show("LightGBM not installed", type="error")
                        return
                    model = LGBMClassifier(verbose=-1)
                else:
                    ui.notification_show("Model not compatible with classification", type="error")
                    return
            else:
                if model_name == "linear regression":
                    model = LinearRegression()
                elif model_name == "ridge":
                    model = Ridge()
                elif model_name == "lasso":
                    model = Lasso()
                elif model_name == "support vector machine":
                    model = SVR()
                elif model_name == "decision tree":
                    model = DecisionTreeRegressor()
                elif model_name == "random forest":
                    model = RandomForestRegressor()
                elif model_name == "stochastic gradient descent":
                    model = SGDRegressor(max_iter=1000)
                elif model_name == "gradient boosting":
                    model = GradientBoostingRegressor()
                elif model_name == "xgboost":
                    if not XGBOOST_AVAILABLE:
                        ui.notification_show("XGBoost not installed", type="error")
                        return
                    model = XGBRegressor()
                elif model_name == "gaussian process":
                    model = GaussianProcessRegressor()
                elif model_name == "lgbm":
                    if not LGBM_AVAILABLE:
                        ui.notification_show("LightGBM not installed", type="error")
                        return
                    model = LGBMRegressor(verbose=-1)
                else:
                    ui.notification_show("Model not compatible with regression", type="error")
                    return
            
            if saved_params:
                try:
                    model.set_params(**saved_params)
                except Exception as e:
                    ui.notification_show(f"Error applying parameters: {str(e)}", type="warning")
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Classification일 경우 확률값도 저장
            y_pred_proba = None
            if target_type == "categorical":
                if hasattr(model, "predict_proba"):
                    y_pred_proba = model.predict_proba(X_test)
            
            if target_type == "categorical":
                metrics = {
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    "F1-Score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
                }
            else:
                metrics = {
                    "MSE": mean_squared_error(y_test, y_pred),
                    "MAE": mean_absolute_error(y_test, y_pred),
                    "R2 Score": r2_score(y_test, y_pred)
                }
            
            trained_model.set({
                'model': model,
                'metrics': metrics,
                'X_test': X_test,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'target_type': target_type,
                'model_name': model_name
            })
            
            ui.notification_show("Model trained successfully!", type="message")
            
        except Exception as e:
            ui.notification_show(f"Error training model: {str(e)}", type="error")
    
    @output
    @render.text
    def model_results():
        model_info = trained_model.get()
        if model_info is None:
            return "No model trained yet"
        
        return f"Model trained successfully!\nTest set size: {len(model_info['y_test'])}"
    
    @output
    @render.table
    def metrics_table():
        model_info = trained_model.get()
        if model_info is None:
            return pd.DataFrame()
        
        metrics = model_info['metrics']
        df_metrics = pd.DataFrame({
            'Metric': list(metrics.keys()),
            'Value': [f"{v:.4f}" for v in metrics.values()]
        })
        return df_metrics
    
    @output
    @render.plot
    def model_viz():
        model_info = trained_model.get()
        if model_info is None:
            return
        
        target_type = model_info.get('target_type')
        y_test = model_info['y_test']
        y_pred = model_info['y_pred']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if target_type == "categorical":
            # ROC Curve for Classification
            y_pred_proba = model_info.get('y_pred_proba')
            
            if y_pred_proba is not None:
                from sklearn.metrics import roc_curve, auc
                from sklearn.preprocessing import label_binarize
                
                # Binary classification
                if len(np.unique(y_test)) == 2:
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
                    roc_auc = auc(fpr, tpr)
                    
                    ax.plot(fpr, tpr, color='darkorange', lw=2, 
                           label=f'ROC curve (AUC = {roc_auc:.2f})')
                    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
                    ax.set_xlim([0.0, 1.0])
                    ax.set_ylim([0.0, 1.05])
                    ax.set_xlabel('False Positive Rate', fontsize=12)
                    ax.set_ylabel('True Positive Rate', fontsize=12)
                    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
                    ax.legend(loc="lower right")
                    ax.grid(True, alpha=0.3)
                else:
                    # Multi-class classification
                    classes = np.unique(y_test)
                    y_test_bin = label_binarize(y_test, classes=classes)
                    
                    for i, cls in enumerate(classes):
                        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
                        roc_auc = auc(fpr, tpr)
                        ax.plot(fpr, tpr, lw=2, label=f'Class {cls} (AUC = {roc_auc:.2f})')
                    
                    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
                    ax.set_xlim([0.0, 1.0])
                    ax.set_ylim([0.0, 1.05])
                    ax.set_xlabel('False Positive Rate', fontsize=12)
                    ax.set_ylabel('True Positive Rate', fontsize=12)
                    ax.set_title('ROC Curves (Multi-class)', fontsize=14, fontweight='bold')
                    ax.legend(loc="lower right", fontsize=8)
                    ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'ROC Curve not available\n(Model does not support probability predictions)', 
                       ha='center', va='center', fontsize=12)
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])
                ax.axis('off')
        else:
            # Residual Plot for Regression
            residuals = y_test - y_pred
            
            ax.scatter(y_pred, residuals, alpha=0.5, edgecolors='k', s=50)
            ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
            ax.set_xlabel('Predicted Values', fontsize=12)
            ax.set_ylabel('Residuals', fontsize=12)
            ax.set_title('Residual Plot', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @output
    @render.plot
    def feature_importance():
        model_info = trained_model.get()
        if model_info is None:
            return
        
        model = model_info['model']
        model_name = model_info.get('model_name', '')
        X_test = model_info['X_test']
        
        # Ensemble 모델 리스트
        ensemble_models = ['random forest', 'gradient boosting', 'xgboost', 'lgbm']
        
        if model_name not in ensemble_models:
            return
        
        # Feature importance 가져오기
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = X_test.columns
            
            # 내림차순 정렬
            indices = np.argsort(importances)[::-1]
            
            # 상위 20개만 표시 (변수가 많을 경우)
            n_features = min(20, len(feature_names))
            indices = indices[:n_features]
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            ax.barh(range(n_features), importances[indices], align='center')
            ax.set_yticks(range(n_features))
            ax.set_yticklabels([feature_names[i] for i in indices])
            ax.invert_yaxis()
            ax.set_xlabel('Importance', fontsize=12)
            ax.set_ylabel('Features', fontsize=12)
            ax.set_title('Feature Importance', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            return fig
        else:
            return

app = App(app_ui, server)