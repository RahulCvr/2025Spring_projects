import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import norm, ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
import yaml
import io

# ============================================================================
# Configuration Loading
# ============================================================================

def load_config():
    """
    Load configuration from YAML file.
    
    Returns:
        dict: Configuration dictionary
        
    Examples:
        >>> config = load_config()
        >>> isinstance(config, dict)
        True
        >>> 'columns' in config
        True
        >>> 'simulation' in config
        True
        >>> 'ui' in config
        True
        >>> all(key in config['columns'] for key in ['date', 'product_id', 'category', 'region'])
        True
    """
    try:
        with open('config.yaml', 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        st.error("Configuration file 'config.yaml' not found!")
        st.stop()
    except yaml.YAMLError as e:
        st.error(f"Error parsing configuration file: {e}")
        st.stop()

# ============================================================================
# Data Loading and Preparation
# ============================================================================

def load_data():
    """
    Load and prepare the retail store inventory data from uploaded CSV file.
    
    Returns:
        pandas.DataFrame: A DataFrame containing the processed retail store inventory data.
        
    Examples:
        >>> # Create a sample DataFrame for testing
        >>> import pandas as pd
        >>> import io
        >>> data = '''Date,Product ID,Category,Region,Weather Condition,Seasonality,Holiday/Promotion,Demand Forecast,Inventory Level,Units Ordered
        ... 2024-01-01,1,Electronics,North,Sunny,Summer,None,100,50,20
        ... 2024-01-02,1,Electronics,North,Sunny,Summer,None,120,30,40'''
        >>> df = pd.read_csv(io.StringIO(data))
        >>> df['Date'] = pd.to_datetime(df['Date'])
        >>> df.sort_values('Date', inplace=True)
        >>> isinstance(df, pd.DataFrame)
        True
        >>> 'Date' in df.columns
        True
        >>> 'Product ID' in df.columns
        True
        >>> df['Date'].dtype == 'datetime64[ns]'
        True
        >>> df['Date'].is_monotonic_increasing
        True
    """
    config = load_config()
    columns = config['columns']
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your retail store inventory data (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # Validate required columns
            required_columns = list(columns.values())
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                st.info("Please ensure your CSV file contains all required columns as specified in the config file.")
                return None
            
            # Process the data
            df[columns['date']] = pd.to_datetime(df[columns['date']])
            df.sort_values(columns['date'], inplace=True)
            
            return df
            
        except Exception as e:
            st.error(f"Error processing the CSV file: {str(e)}")
            return None
    else:
        st.info("Please upload a CSV file to begin.")
        return None

# ============================================================================
# UI and Parameter Collection
# ============================================================================

def validate_numeric_input(value, field_name, min_value=0.01):
    """
    Validate numeric input values.
    
    Args:
        value: Input value to validate
        field_name: Name of the field for error message
        min_value: Minimum allowed value (default: 0.01)
        
    Returns:
        float: Validated value
        
    Examples:
        >>> validate_numeric_input(1.5, "Test Field")
        1.5
        >>> validate_numeric_input(0.5, "Test Field")
        0.5
        >>> validate_numeric_input(0, "Test Field")  # doctest: +SKIP
        Traceback (most recent call last):
        ...
        >>> validate_numeric_input("abc", "Test Field")  # doctest: +SKIP
        Traceback (most recent call last):
        ...
    """
    try:
        value = float(value)
        if value < min_value:
            st.error(f"{field_name} must be greater than {min_value}")
            st.stop()
        return value
    except (ValueError, TypeError):
        st.error(f"{field_name} must be a valid number")
        st.stop()

def setup_ui():
    """
    Set up the Streamlit UI and collect user input parameters.
    
    Returns:
        tuple: A tuple containing all user input parameters
    
    Note:
        This function is designed to be run in a Streamlit context and cannot be tested
        directly with doctest.
    """
    config = load_config()
    columns = config['columns']
    sim_config = config['simulation']
    ui_config = config['ui']
    
    st.title(ui_config['title'])
    st.sidebar.header(ui_config['sidebar_header'])
    
    # Load and prepare data
    df = load_data()
    if df is None:
        st.stop()
        
    products = sorted(df[columns['product_id']].unique())

    # User input parameters
    product_id = st.sidebar.selectbox("Select Product", products)
    
    service_level = st.sidebar.slider(
        "Service Level (%)",
        sim_config['service_level_range']['min'],
        sim_config['service_level_range']['max'],
        sim_config['service_level_range']['default']
    ) / 100
    
    holding_cost = validate_numeric_input(
        st.sidebar.number_input(
            "Holding Cost per Unit per Day",
            value=sim_config['holding_cost_default'],
            min_value=0.01,
            step=0.01
        ),
        "Holding Cost"
    )
    
    stockout_cost = validate_numeric_input(
        st.sidebar.number_input(
            "Stockout Cost per Unit",
            value=sim_config['stockout_cost_default'],
            min_value=0.01,
            step=0.01
        ),
        "Stockout Cost"
    )
    
    ordering_cost = validate_numeric_input(
        st.sidebar.number_input(
            "Fixed Cost per Order",
            value=sim_config['ordering_cost_default'],
            min_value=0.01,
            step=0.01
        ),
        "Ordering Cost"
    )
    
    reorder_point_fixed = validate_numeric_input(
        st.sidebar.slider(
            "Fixed Reorder Point",
            sim_config['reorder_point_range']['min'],
            sim_config['reorder_point_range']['max'],
            sim_config['reorder_point_range']['default'],
            step=sim_config['reorder_point_range']['step']
        ),
        "Reorder Point"
    )
    
    fixed_order_quantity = validate_numeric_input(
        st.sidebar.number_input(
            "Manual Order Quantity (Fixed Policy)",
            value=sim_config['fixed_order_quantity_default'],
            min_value=1,
            step=1
        ),
        "Order Quantity"
    )
    
    monthly_safety_stock = validate_numeric_input(
        st.sidebar.number_input(
            "Uniform Safety Stock for Monthly Test (units)",
            min_value=0,
            value=sim_config['monthly_safety_stock_default'],
            step=1,
            key='monthly_safety_stock'
        ),
        "Safety Stock",
        min_value=0
    )
    
    monthly_target_fill = st.sidebar.slider(
        "Target Fill Rate for Monthly Test",
        sim_config['monthly_target_fill_range']['min'],
        sim_config['monthly_target_fill_range']['max'],
        sim_config['monthly_target_fill_range']['default'],
        key='monthly_target_fill'
    )
    
    return (df, product_id, service_level, holding_cost, stockout_cost, 
            ordering_cost, reorder_point_fixed, fixed_order_quantity, 
            monthly_safety_stock, monthly_target_fill)

# ============================================================================
# Model Preparation and Prediction
# ============================================================================

def prepare_model_and_predictions(df, product_id):
    """
    Prepare the regression model and generate predictions for the selected product.
    
    Args:
        df (pandas.DataFrame): The full dataset.
        product_id: Selected product ID.
        
    Returns:
        tuple: (filtered_df, predicted_all, residual_std)
        
    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> data = {'Date': pd.date_range(start='2024-01-01', periods=5), 'Product ID': [1, 1, 1, 1, 1], 'Category': ['Electronics'] * 5, 'Region': ['North'] * 5, 'Weather Condition': ['Sunny'] * 5, 'Seasonality': ['Summer'] * 5, 'Holiday/Promotion': ['None'] * 5, 'Demand Forecast': [100, 120, 150, 110, 130], 'Inventory Level': [50, 30, 40, 20, 60], 'Units Ordered': [20, 40, 30, 50, 10], 'Store ID': [1, 1, 1, 1, 1]}
        >>> df = pd.DataFrame(data)
        >>> filtered_df, predicted_all, residual_std = prepare_model_and_predictions(df, 1)
        >>> isinstance(filtered_df, pd.DataFrame)
        True
        >>> isinstance(predicted_all, np.ndarray)
        True
        >>> isinstance(residual_std, float)
        True
    """
    config = load_config()
    columns = config['columns']
    
    # Filter data for selected product
    filtered_df = df[df[columns['product_id']] == product_id].copy()

    # Prepare regression model
    full_df = df.copy()
    cat_cols = [
        columns['category'],
        columns['region'],
        columns['weather_condition'],
        columns['seasonality'],
        columns['holiday_promotion'],
        columns['product_id']
    ]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        full_df[col] = le.fit_transform(full_df[col])
        encoders[col] = le

    X_full = full_df.drop(columns=[
        columns['demand_forecast'],
        columns['date'],
        columns['store_id'],
        columns['inventory_level'],
        columns['units_ordered']
    ])
    y_full = full_df[columns['demand_forecast']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)
    reg_model = LinearRegression()
    reg_model.fit(X_scaled, y_full)

    # Prepare predictions
    filtered = filtered_df.copy()
    for col in cat_cols:
        filtered[col] = encoders[col].transform(filtered[col])
    X_filt = filtered.drop(columns=[
        columns['demand_forecast'],
        columns['date'],
        columns['store_id'],
        columns['inventory_level'],
        columns['units_ordered']
    ])
    X_filt_scaled = scaler.transform(X_filt)
    predicted_all = reg_model.predict(X_filt_scaled)
    residual_std = np.std(y_full - reg_model.predict(X_scaled))

    return filtered_df, predicted_all, residual_std

# ============================================================================
# Experiment-1: Effect of Dynamic vs Fixed Reorder Policy
# ============================================================================

def find_optimal_order_quantity(predicted_all, residual_std, z_score, n_simulations, n_days,
                              holding_cost, stockout_cost, ordering_cost, reorder_point_fixed):
    """
    Calculate the optimal order quantity based on demand predictions and cost parameters.
    
    Args:
        predicted_all (numpy.ndarray): Array of predicted daily demands
        residual_std (float): Standard deviation of prediction residuals
        z_score (float): Z-score corresponding to desired service level
        n_simulations (int): Number of simulation runs
        n_days (int): Number of days to simulate
        holding_cost (float): Cost per unit per day for holding inventory
        stockout_cost (float): Cost per unit for stockouts
        ordering_cost (float): Fixed cost per order
        reorder_point_fixed (float): Fixed reorder point for comparison
        
    Returns:
        tuple: (optimal_quantity, average_cost)
        
    Examples:
        >>> predicted = np.array([10, 12, 15, 11, 13])
        >>> residual_std = 2.0
        >>> z_score = 1.96
        >>> n_sim = 10
        >>> n_days = 30
        >>> holding = 0.05
        >>> stockout = 2.0
        >>> ordering = 50.0
        >>> reorder = 100
        >>> opt_q, avg_cost = find_optimal_order_quantity(predicted, residual_std, z_score, n_sim, n_days, holding, stockout, ordering, reorder)
        >>> isinstance(opt_q, int)
        True
        >>> isinstance(avg_cost, float)
        True
        >>> opt_q > 0
        True
        >>> avg_cost >= 0
        True
    """
    mean_daily_demand = np.mean(predicted_all)
    std_daily_demand = residual_std

    mean_weekly_demand = 7 * mean_daily_demand
    std_weekly_demand = np.sqrt(7) * std_daily_demand

    optimal_quantity = int(np.round(mean_weekly_demand + z_score * std_weekly_demand))
    df_sim = run_simulation(True, optimal_quantity, predicted_all, residual_std, n_simulations, n_days,
                          holding_cost, stockout_cost, ordering_cost, z_score, reorder_point_fixed)
    average_cost = df_sim["Total Cost"].mean()

    return optimal_quantity, average_cost

def run_simulation(dynamic, order_quantity, predicted_all, residual_std, n_simulations, n_days, 
                  holding_cost, stockout_cost, ordering_cost, z_score, reorder_point_fixed):
    """
    Run inventory simulation with either dynamic or fixed reorder points.
    
    Args:
        dynamic (bool): If True, use dynamic reorder points; if False, use fixed reorder point
        order_quantity (int): Quantity to order when reorder point is reached
        predicted_all (numpy.ndarray): Array of predicted daily demands
        residual_std (float): Standard deviation of prediction residuals
        n_simulations (int): Number of simulation runs
        n_days (int): Number of days to simulate
        holding_cost (float): Cost per unit per day for holding inventory
        stockout_cost (float): Cost per unit for stockouts
        ordering_cost (float): Fixed cost per order
        z_score (float): Z-score corresponding to desired service level
        reorder_point_fixed (float): Fixed reorder point (used when dynamic=False)
        
    Returns:
        pandas.DataFrame: DataFrame containing simulation results
        
    Examples:
        >>> predicted = np.array([10, 12, 15, 11, 13])
        >>> residual_std = 2.0
        >>> z_score = 1.96
        >>> n_sim = 2
        >>> n_days = 5
        >>> holding = 0.05
        >>> stockout = 2.0
        >>> ordering = 50.0
        >>> reorder = 100
        >>> df = run_simulation(True, 50, predicted, residual_std, n_sim, n_days, holding, stockout, ordering, z_score, reorder)
        >>> isinstance(df, pd.DataFrame)
        True
        >>> all(col in df.columns for col in ['Total Cost', 'Holding', 'Stockout', 'Ordering', 'Stockout Days'])
        True
        >>> len(df) == n_sim
        True
        >>> all(df['Total Cost'] >= 0)
        True
    """
    results = []
    for _ in range(n_simulations):
        inventory = order_quantity
        on_order = []
        total_holding, total_stockout, total_ordering, stockout_days = 0, 0, 0, 0

        sample_indices = np.random.choice(len(predicted_all), size=n_days)
        sampled_pred = predicted_all[sample_indices]

        for i in range(n_days):
            demand = max(0, np.random.normal(sampled_pred[i], residual_std))
            on_order = [[q, lt-1] for q, lt in on_order if lt > 1]
            arrivals = sum(q for q, lt in on_order if lt <= 1)
            inventory += arrivals

            if inventory >= demand:
                inventory -= demand
                stockout = 0
            else:
                stockout = demand - inventory
                inventory = 0
                stockout_days += 1

            total_holding += inventory * holding_cost
            total_stockout += stockout * stockout_cost

            lead_time = np.random.triangular(1, 2, 3)
            reorder_point = (sampled_pred[i] * lead_time + z_score * residual_std * np.sqrt(lead_time)) if dynamic else reorder_point_fixed
            inv_pos = inventory + sum(q for q, _ in on_order)

            if inv_pos < reorder_point:
                on_order.append([order_quantity, lead_time])
                total_ordering += ordering_cost

        total_cost = total_holding + total_stockout + total_ordering
        results.append({
            "Total Cost": total_cost,
            "Holding": total_holding,
            "Stockout": total_stockout,
            "Ordering": total_ordering,
            "Stockout Days": stockout_days
        })
    return pd.DataFrame(results)

def run_simulations(predicted_all, residual_std, z_score, n_simulations, n_days,
                   holding_cost, stockout_cost, ordering_cost, reorder_point_fixed, fixed_order_quantity):
    """
    Run both fixed and dynamic policy simulations.
    
    Returns:
        tuple: (optimal_q, optimal_cost, df_fixed, df_dynamic)
    """
    with st.spinner("Finding optimal order quantity and running simulations..."):
        optimal_q, optimal_cost = find_optimal_order_quantity(
            predicted_all, residual_std, z_score, n_simulations, n_days,
            holding_cost, stockout_cost, ordering_cost, reorder_point_fixed
        )
        st.sidebar.write(f"**Optimal Order Quantity (Dynamic):** {optimal_q}")

        df_fixed = run_simulation(False, fixed_order_quantity, predicted_all, residual_std, n_simulations, n_days,
                                holding_cost, stockout_cost, ordering_cost, z_score, reorder_point_fixed)
        df_dynamic = run_simulation(True, optimal_q, predicted_all, residual_std, n_simulations, n_days,
                                  holding_cost, stockout_cost, ordering_cost, z_score, reorder_point_fixed)
    
    return optimal_q, optimal_cost, df_fixed, df_dynamic

def display_simulation_results(df_fixed, df_dynamic):
    """
    Display simulation results and statistical analysis.
    """
    st.header("Experiment-1: Effect of Dynamic vs Fixed Reorder Policy")
    st.subheader("Simulation Results")
    st.markdown("**Fixed Policy**")
    st.dataframe(df_fixed.describe())

    st.markdown("**Dynamic Policy**")
    st.dataframe(df_dynamic.describe())

    # Visualize cost distributions
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.kdeplot(df_fixed["Total Cost"], label="Fixed", fill=True, ax=ax)
    sns.kdeplot(df_dynamic["Total Cost"], label="Dynamic", fill=True, ax=ax)
    ax.axvline(df_fixed["Total Cost"].mean(), linestyle="--", color="blue")
    ax.axvline(df_dynamic["Total Cost"].mean(), linestyle="--", color="orange")
    ax.set_title("Cost Distribution Comparison")
    ax.set_xlabel("Total Cost")
    ax.legend()
    st.pyplot(fig)

    # Statistical analysis
    stat, pval = ttest_ind(df_fixed["Total Cost"], df_dynamic["Total Cost"], equal_var=False)
    st.write(f"**T-test p-value:** {pval}")

    if pval < 0.05:
        st.warning("Reject Null Hypothesis: Dynamic policy significantly reduces cost (p < 0.05)")
    else:
        st.success("Failed to Reject Null Hypothesis: No significant difference found at 95% confidence level")

# ============================================================================
# Experiment-2: Effect of Seasonal Variance on Service Level Performance
# ============================================================================

def perform_seasonal_analysis(filtered_df, monthly_safety_stock, monthly_target_fill):
    """
    Perform and display seasonal analysis of inventory performance.
    
    Returns:
        tuple: (monthly_results, monthly_samples)
    """
    config = load_config()
    columns = config['columns']
    
    st.header("Experiment-2: Effect of Seasonal Variance on Service Level Performance")
    season_names = list(filtered_df[columns['seasonality']].dropna().unique())
    monthly_results = {}
    monthly_samples = {}

    # Calculate monthly fill rates
    for season in season_names:
        season_df = filtered_df[filtered_df[columns['seasonality']] == season].copy()
        if len(season_df) == 0:
            monthly_results[season] = None
            monthly_samples[season] = None
            continue
        season_df['YearMonth'] = season_df[columns['date']].dt.to_period('M')
        months = season_df['YearMonth'].unique()
        fill_sample = []
        for month in months:
            month_df = season_df[season_df['YearMonth'] == month]
            n_days = len(month_df)
            forecast_sum = month_df[columns['demand_forecast']].sum()
            errors = (month_df[columns['units_sold']] - month_df[columns['demand_forecast']]).values
            n_trials = 1000
            for _ in range(n_trials):
                simulated_demand = forecast_sum + np.sum(np.random.choice(errors, size=n_days, replace=True))
                inventory = forecast_sum + monthly_safety_stock
                if simulated_demand > inventory:
                    fill_sample.append(0)
                else:
                    fill_sample.append(1)
        fill_rate = np.mean(fill_sample) if fill_sample else None
        monthly_results[season] = fill_rate
        monthly_samples[season] = fill_sample

    # Display monthly results
    st.write("**Monthly Aggregated Fill Rates with Uniform Safety Stock:**")
    for season in season_names:
        fill = monthly_results[season]
        if fill is not None:
            st.write(f"{season}: {fill:.2%} {'✅' if fill >= monthly_target_fill else '❌'}")
        else:
            st.write(f"{season}: No data")

    shortfall = [s for s, f in monthly_results.items() if f is not None and f < monthly_target_fill]
    if shortfall:
        st.warning(f"Seasons below target fill rate ({monthly_target_fill:.0%}): {', '.join(shortfall)}")
    else:
        st.success("All seasons meet or exceed the target fill rate!")

    # ANOVA analysis
    anova_data = [v for v in monthly_samples.values() if v is not None and len(v) > 0]
    if len(anova_data) > 1:
        f_stat, p_value = f_oneway(*anova_data)
        st.write(f"**ANOVA F-statistic:** {f_stat:.2f}")
        st.write(f"**p-value:** {p_value:.4g}")
        if p_value < 0.05:
            st.warning("Reject Null Hypothesis: There is a significant difference in service levels between seasons.")
        else:
            st.success("Failed to Reject Null Hypothesis: No significant difference in service levels between seasons.")
    else:
        st.info("Not enough data for ANOVA test.")

    # Visualize seasonal fill rates
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(season_names, [monthly_results[s] if monthly_results[s] is not None else 0 for s in season_names])
    ax.axhline(y=monthly_target_fill, color='r', linestyle='--', label='Target Fill Rate')
    ax.set_ylabel('Fill Rate')
    ax.set_title('Monthly Aggregated Seasonal Fill Rates')
    ax.legend()
    st.pyplot(fig)

    return monthly_results, monthly_samples

def calculate_safety_stock_recommendations(filtered_df, monthly_target_fill):
    """
    Calculate and display recommended safety stock levels for each season.
    
    Returns:
        dict: Dictionary containing minimum safety stock recommendations for each season
    """
    config = load_config()
    columns = config['columns']
    
    st.subheader("Recommended Minimum Safety Stock per Season (Monthly Aggregation)")
    season_names = list(filtered_df[columns['seasonality']].dropna().unique())
    min_safety_stock = {}
    
    for season in season_names:
        season_df = filtered_df[filtered_df[columns['seasonality']] == season].copy()
        if len(season_df) == 0:
            min_safety_stock[season] = None
            continue
        season_df['YearMonth'] = season_df[columns['date']].dt.to_period('M')
        months = season_df['YearMonth'].unique()
        found = False
        for ss in range(0, 51):
            fill_sample = []
            for month in months:
                month_df = season_df[season_df['YearMonth'] == month]
                n_days = len(month_df)
                forecast_sum = month_df[columns['demand_forecast']].sum()
                errors = (month_df[columns['units_sold']] - month_df[columns['demand_forecast']]).values
                n_trials = 200
                for _ in range(n_trials):
                    simulated_demand = forecast_sum + np.sum(np.random.choice(errors, size=n_days, replace=True))
                    inventory = forecast_sum + ss
                    if simulated_demand > inventory:
                        fill_sample.append(0)
                    else:
                        fill_sample.append(1)
            fill_rate = np.mean(fill_sample) if fill_sample else 0
            if fill_rate >= monthly_target_fill:
                min_safety_stock[season] = ss
                found = True
                break
        if not found:
            min_safety_stock[season] = ">50"

    # Display safety stock recommendations
    st.write("**Minimum Safety Stock Needed per Season to Meet Target Fill Rate:**")
    for season in season_names:
        st.write(f"{season}: {min_safety_stock[season]} units")
    
    return min_safety_stock

# ============================================================================
# Main App
# ============================================================================

def main():
    """
    Main function to run the Inventory Policy Simulator application.
    """
    # Setup UI and get parameters
    (df, product_id, service_level, holding_cost, stockout_cost, 
     ordering_cost, reorder_point_fixed, fixed_order_quantity, 
     monthly_safety_stock, monthly_target_fill) = setup_ui()

    # Prepare model and predictions
    filtered_df, predicted_all, residual_std = prepare_model_and_predictions(df, product_id)

    # Simulation parameters
    n_days = 365
    n_simulations = 300
    z_score = norm.ppf(service_level)

    # Run simulations
    optimal_q, optimal_cost, df_fixed, df_dynamic = run_simulations(
        predicted_all, residual_std, z_score, n_simulations, n_days,
        holding_cost, stockout_cost, ordering_cost, reorder_point_fixed, fixed_order_quantity
    )

    # Display results
    display_simulation_results(df_fixed, df_dynamic)

    # Perform seasonal analysis
    monthly_results, monthly_samples = perform_seasonal_analysis(
        filtered_df, monthly_safety_stock, monthly_target_fill
    )

    # Calculate and display safety stock recommendations
    min_safety_stock = calculate_safety_stock_recommendations(filtered_df, monthly_target_fill)

if __name__ == "__main__":
    main() 