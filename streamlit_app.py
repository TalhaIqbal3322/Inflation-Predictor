import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

class InflationPredictor:
    def __init__(self):
        """Initialize the Inflation Predictor with preprocessing tools"""
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.model = None
        self.feature_names = None
        self.inflation_column = None

    def identify_inflation_column(self, data):
        """
        Attempt to identify the inflation column
        Looks for columns with 'inflation', 'price', or 'rate' in their name (case-insensitive)
        """
        # List of potential column name patterns
        inflation_patterns = ['inflation', 'price', 'rate', 'cpi']
        
        # Find matching columns
        matching_cols = [
            col for col in data.columns 
            if any(pattern in col.lower() for pattern in inflation_patterns)
        ]
        
        # If exact match found, return the first match
        if matching_cols:
            return matching_cols[0]
        
        # If no match, raise an informative error
        raise ValueError(f"""
        No inflation-related column found. 
        Available columns: {list(data.columns)}
        
        Tip: Ensure your dataset has a column related to inflation, 
        such as 'Inflation', 'InflationRate', 'PriceIndex', etc.
        """)

    def preprocess_data(self, data):
        """Preprocess the input data for modeling"""
        # Validate input
        if data is None or data.empty:
            raise ValueError("Input data is empty or None")

        # Identify inflation column
        try:
            self.inflation_column = self.identify_inflation_column(data)
        except ValueError as e:
            st.error(str(e))
            raise

        # Select numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        # Ensure we have enough features
        if len(numeric_cols) < 2:
            raise ValueError("Not enough numeric columns for prediction")

        # Separate features and target
        X = data[numeric_cols].drop(columns=[self.inflation_column])
        y = data[self.inflation_column]
        
        # Remove rows with NaN in the target variable
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        # Store feature names for later use
        self.feature_names = X.columns.tolist()
        
        # Impute and scale features
        X_imputed = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        return X_scaled, y

    def train_model(self, X, y, model_type='rf'):
        """Train the prediction model"""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Choose model
        if model_type == 'rf':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)

        # Train the model
        self.model.fit(X_train, y_train)

        # Make predictions
        y_pred = self.model.predict(X_test)

        # Calculate metrics
        metrics = {
            'MSE': mean_squared_error(y_test, y_pred),
            'MAE': mean_absolute_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred),
            'MAPE': mean_absolute_percentage_error(y_test, y_pred) * 100
        }

        return metrics, y_test, y_pred

    def predict_future(self, X, future_years):
        """Predict future inflation"""
        # If more years than available prediction points, repeat the last prediction
        if len(future_years) > len(X):
            # Repeat the last row of X to match the number of years
            additional_rows_needed = len(future_years) - len(X)
            last_row = X[-1:].repeat(additional_rows_needed, axis=0)
            X_extended = np.vstack([X, last_row])
        else:
            X_extended = X[:len(future_years)]
        
        # Predict inflation
        predictions = self.model.predict(X_extended)
        
        # Calculate confidence interval
        std_dev = np.std(predictions)
        confidence_interval = 1.96 * std_dev
        
        pred_df = pd.DataFrame({
            'Year': future_years[:len(predictions)],
            'Predicted Inflation (%)': predictions,
            'Lower Bound (%)': predictions - confidence_interval,
            'Upper Bound (%)': predictions + confidence_interval
        })
        
        return pred_df

    def get_feature_importance(self):
        """Get feature importances if model supports it"""
        if hasattr(self.model, 'feature_importances_'):
            return pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False)
        return None

def main():
    st.set_page_config(page_title="Inflation Predictor", layout="wide")
    
    # Custom CSS
    st.markdown("""
    <style>
    .reportview-container {
        background: #F0F2F6;
    }
    .sidebar .sidebar-content {
        background: #FFFFFF;
    }
    .stAlert {
        background-color: #f8d7da;
        color: #721c24;
        border-color: #f5c6cb;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize predictor
    predictor = InflationPredictor()

    # Sidebar navigation
    st.sidebar.title("🌍 Inflation Predictor")
    page = st.sidebar.radio("Navigate", 
        ["Data Upload", "Inflation Forecast", "Model Insights", "Economic Analysis", "About"])

    if page == "About":
        st.header("📖 About the Inflation Predictor")
        
        # Project Overview
        st.subheader("Project Overview")
        st.markdown("""
        The Inflation Predictor is an advanced machine learning application designed to forecast inflation rates 
        using economic indicators. By leveraging sophisticated predictive models like Random Forest and 
        Gradient Boosting, the application provides comprehensive insights into potential future economic trends.
        """)
        
        # Developers Section
        st.subheader("Developed By")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Talha Iqbal**")
            st.markdown("**Student of Artificial Intelligence At Air University Islamabad**")
            
           
        
   
        
        # Technical Details
       
        st.markdown("""
           ### Advanced Economic Forecasting Platform 
           **Key Features:** 
           - Machine Learning Inflation Predictions 
           - Multiple Model Support 
           - Hyperparameter Optimization 
           - Confidence Interval Forecasting 
                    
           **Predictive Models**
           - Random Forest Regressor
           - Gradient Boosting Regressor
            

           **Technologies:** 
           - Python 
           - Streamlit 
           - Scikit-Learn 
           - Optuna 
           - Plotly 

            **How to Use:**
           1. Upload your economic dataset
           2. Explore data distribution
           3. Generate inflation forecasts
           4. Analyze model performance and insights
       """)
        
        # Methodology
        st.subheader("Methodology")
        st.markdown("""
        The inflation prediction model employs a robust machine learning pipeline:
        1. **Data Preprocessing**: 
           - Handles missing values
           - Scales numerical features
           - Identifies relevant economic indicators
        
        2. **Model Training**:
           - Splits data into training and testing sets
           - Trains ensemble models for robust predictions
           - Evaluates model performance using multiple metrics
        
        3. **Forecasting**:
           - Predicts future inflation rates
           - Calculates confidence intervals
           - Provides detailed economic insights
        """)
        
       
 

    if page == "Data Upload":
        st.header("📂 Upload Economic Dataset")
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.write("Dataset Overview:")
                st.dataframe(data.head())
                
                # Display basic dataset info
                st.write("Dataset Information:")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Rows", len(data))
                    st.metric("Numeric Columns", len(data.select_dtypes(include=[np.number]).columns))
                with col2:
                    st.write("Columns:", ", ".join(data.columns))
                
                # Attempt to identify inflation column
                try:
                    inflation_col = predictor.identify_inflation_column(data)
                    st.success(f"Inflation column identified: {inflation_col}")
                except ValueError as e:
                    st.warning(str(e))
                
                # Store data for other pages
                st.session_state.uploaded_data = data
            
            except Exception as e:
                st.error(f"Error uploading file: {e}")

    elif page == "Inflation Forecast":
        st.header("📈 Inflation Forecast")
        
        # Check if data is uploaded
        if 'uploaded_data' not in st.session_state:
            st.warning("Please upload a dataset first in the 'Data Upload' section.")
            return

        data = st.session_state.uploaded_data

        try:
            # Preprocess data
            X, y = predictor.preprocess_data(data)

            # Determine last year in the dataset for reference
            try:
                last_year = int(data.columns[data.columns.str.contains('year', case=False)].values[0])
            except:
                last_year = 2023  # Default fallback

            # Model and forecast configuration
            col1, col2 = st.columns(2)
            with col1:
                model_type = st.selectbox("Select Model", ["Random Forest", "Gradient Boosting"])
            with col2:
                # Limited prediction years based on last known year
                future_years = st.multiselect(
                    "Select Prediction Years", 
                    options=[last_year + i for i in range(1, 6)],  # Next 5 years
                    default=[last_year + 1, last_year + 2]  # Default to next 2 years
                )

            if st.button("Generate Forecast"):
                # Determine model type
                model_str = 'rf' if model_type == "Random Forest" else 'gb'
                
                # Train model
                with st.spinner('Training Model...'):
                    metrics, y_test, y_pred = predictor.train_model(X, y, model_type=model_str)
                
                # Display model metrics
                st.subheader("Model Performance")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mean Absolute Error", f"{metrics['MAE']:.4f}")
                    st.metric("R² Score", f"{metrics['R2']:.4f}")
                with col2:
                    st.metric("Mean Squared Error", f"{metrics['MSE']:.4f}")
                    st.metric("Mean Absolute % Error", f"{metrics['MAPE']:.4f}%")
                
                # Predict future inflation
                predictions = predictor.predict_future(X, future_years)
                
                # Visualization
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=predictions['Year'], 
                    y=predictions['Predicted Inflation (%)'], 
                    mode='lines+markers',
                    name='Predicted Inflation',
                    line=dict(color='blue'),
                    text=[f'{y}: {val:.2f}%' for y, val in zip(predictions['Year'], predictions['Predicted Inflation (%)'])],
                    hoverinfo='text'
                ))
                fig.add_trace(go.Scatter(
                    x=predictions['Year'], 
                    y=predictions['Lower Bound (%)'], 
                    mode='lines', 
                    name='Lower Bound',
                    line=dict(color='red', dash='dot')
                ))
                fig.add_trace(go.Scatter(
                    x=predictions['Year'], 
                    y=predictions['Upper Bound (%)'], 
                    mode='lines', 
                    name='Upper Bound',
                    line=dict(color='green', dash='dot')
                ))
                fig.update_layout(
                    title='Inflation Rate Forecast',
                    xaxis_title='Year',
                    yaxis_title='Inflation Rate (%)'
                )
                st.plotly_chart(fig)
                
                # Enhanced predictions display
                st.subheader("Detailed Inflation Predictions")
                # Create a more readable table with formatting
                formatted_predictions = predictions.copy()
                formatted_predictions['Predicted Inflation (%)'] = formatted_predictions['Predicted Inflation (%)'].apply(lambda x: f'{x:.2f}%')
                formatted_predictions['Lower Bound (%)'] = formatted_predictions['Lower Bound (%)'].apply(lambda x: f'{x:.2f}%')
                formatted_predictions['Upper Bound (%)'] = formatted_predictions['Upper Bound (%)'].apply(lambda x: f'{x:.2f}%')
                
                st.table(formatted_predictions)
                
                # Key insights
                max_inflation = predictions['Predicted Inflation (%)'].max()
                min_inflation = predictions['Predicted Inflation (%)'].min()
                avg_inflation = predictions['Predicted Inflation (%)'].mean()
                
                st.subheader("Inflation Forecast Insights")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Maximum Predicted Inflation", f"{max_inflation:.2f}%")
                with col2:
                    st.metric("Minimum Predicted Inflation", f"{min_inflation:.2f}%")
                with col3:
                    st.metric("Average Predicted Inflation", f"{avg_inflation:.2f}%")

        except Exception as e:
            st.error(f"Error in forecast generation: {e}")

    elif page == "Model Insights":
        st.header("🧠 Model Insights")
        
        # Check if data is uploaded
        if 'uploaded_data' not in st.session_state:
            st.warning("Please upload a dataset first in the 'Data Upload' section.")
            return

        data = st.session_state.uploaded_data

        try:
            # Preprocess data
            X, y = predictor.preprocess_data(data)

            # Train model
            metrics, y_test, y_pred = predictor.train_model(X, y)

            # Display model metrics
            st.subheader("Model Performance Metrics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mean Absolute Error", f"{metrics['MAE']:.4f}")
                st.metric("R² Score", f"{metrics['R2']:.4f}")
            with col2:
                st.metric("Mean Squared Error", f"{metrics['MSE']:.4f}")
                st.metric("Mean Absolute % Error", f"{metrics['MAPE']:.4f}%")

            # Feature Importance
            feature_importance = predictor.get_feature_importance()
            if feature_importance is not None:
                st.subheader("Feature Importance")
                
                # Bar plot
                fig = px.bar(
                    feature_importance, 
                    x='Importance', 
                    y='Feature', 
                    orientation='h',
                    title='Feature Importance in Inflation Prediction'
                )
                st.plotly_chart(fig)
                
                # Detailed table
                st.dataframe(feature_importance)

            # Residual Analysis
            residuals = y_test - y_pred
            fig_residuals = go.Figure()
            fig_residuals.add_trace(go.Scatter(
                x=y_pred, 
                y=residuals, 
                mode='markers',
                name='Residuals'
            ))
            fig_residuals.update_layout(
                title='Residual Plot',
                xaxis_title='Predicted Values',
                yaxis_title='Residuals'
            )
            st.plotly_chart(fig_residuals)

        except Exception as e:
            st.error(f"Error in generating model insights: {e}")

    elif page == "Economic Analysis":
        st.header("🌐 Economic Indicator Analysis")
        
        # Check if data is uploaded
        if 'uploaded_data' not in st.session_state:
            st.warning("Please upload a dataset first in the 'Data Upload' section.")
            return

        data = st.session_state.uploaded_data

        # Select numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        # Correlation Matrix
        correlation_matrix = data[numeric_cols].corr()
        
        # Correlation Heatmap
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu_r',
            zmin=-1, 
            zmax=1
        ))
        fig.update_layout(title='Economic Indicators Correlation Heatmap')
        st.plotly_chart(fig)

        # Detailed Correlation with Inflation Column
        try:
            inflation_col = predictor.identify_inflation_column(data)
            st.subheader(f"Correlation with {inflation_col}")
            inflation_correlations = correlation_matrix[inflation_col].sort_values(ascending=False)
            st.dataframe(inflation_correlations)
        except ValueError as e:
            st.error(str(e))

if __name__ == "__main__":
    main()

