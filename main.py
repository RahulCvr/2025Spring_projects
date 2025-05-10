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

# NOTE: Will add detailed docstrings and comments to the code soon.

# ============================================================================
# Data Loading and Preparation
# ============================================================================

@st.cache_data
def load_data():
    df = pd.read_csv("retail_store_inventory.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)
    return df

# ============================================================================
# Simulation Helper Functions
# ============================================================================

def find_optimal_order_quantity(predicted_all, residual_std, z_score, n_simulations, n_days,
                              holding_cost, stockout_cost, ordering_cost, reorder_point_fixed):
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

# ============================================================================
# Main Application
# ============================================================================

def main():
    # Initialize UI
    st.title("Inventory Policy Simulator: Fixed vs Dynamic - Weekly Ordering")
    st.sidebar.header("Simulation Settings")

    # Load and prepare data
    df = load_data()
    stores = sorted(df["Store ID"].unique())
    products = sorted(df["Product ID"].unique())

    # User input parameters
    store_id = st.sidebar.selectbox("Select Store", stores)
    product_id = st.sidebar.selectbox("Select Product", products)
    service_level = st.sidebar.slider("Service Level (%)", 80, 99, 95) / 100
    holding_cost = st.sidebar.number_input("Holding Cost per Unit per Day", value=0.05)
    stockout_cost = st.sidebar.number_input("Stockout Cost per Unit", value=2.00)
    ordering_cost = st.sidebar.number_input("Fixed Cost per Order", value=50.00)
    reorder_point_fixed = st.sidebar.slider("Fixed Reorder Point", 50, 300, 150, step=10)
    fixed_order_quantity = st.sidebar.number_input("Manual Order Quantity (Fixed Policy)", value=300)
    monthly_safety_stock = st.sidebar.number_input(
        "Uniform Safety Stock for Monthly Test (units)", min_value=0, value=20, key='monthly_safety_stock'
    )
    monthly_target_fill = st.sidebar.slider(
        "Target Fill Rate for Monthly Test", 0.80, 0.99, 0.95, key='monthly_target_fill'
    )

    # Filter data for selected store and product
    filtered_df = df[(df["Store ID"] == store_id) & (df["Product ID"] == product_id)].copy()

    # Prepare regression model
    full_df = df.copy()
    cat_cols = ["Category", "Region", "Weather Condition", "Seasonality", "Holiday/Promotion", "Product ID"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        full_df[col] = le.fit_transform(full_df[col])
        encoders[col] = le

    X_full = full_df.drop(columns=['Demand Forecast', 'Date', 'Store ID', 'Inventory Level', 'Units Ordered'])
    y_full = full_df['Demand Forecast']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)
    reg_model = LinearRegression()
    reg_model.fit(X_scaled, y_full)

    # Prepare predictions
    filtered = filtered_df.copy()
    for col in cat_cols:
        filtered[col] = encoders[col].transform(filtered[col])
    X_filt = filtered.drop(columns=['Demand Forecast', 'Date', 'Store ID', 'Inventory Level', 'Units Ordered'])
    X_filt_scaled = scaler.transform(X_filt)
    predicted_all = reg_model.predict(X_filt_scaled)
    residual_std = np.std(y_full - reg_model.predict(X_scaled))

    # Simulation parameters
    n_days = 365
    n_simulations = 300
    z_score = norm.ppf(service_level)

    # Run simulations
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

    # Display simulation results
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
        st.success("Hypothesis Rejected: Dynamic policy significantly reduces cost (p < 0.05)")
    else:
        st.warning("Fail to Reject Null: No significant difference at 95% confidence level")

    # Seasonal analysis
    st.subheader("Effect of Seasonal Variance on Service Level Performance")
    season_names = list(filtered_df['Seasonality'].dropna().unique())
    monthly_results = {}
    monthly_samples = {}

    # Calculate monthly fill rates
    for season in season_names:
        season_df = filtered_df[filtered_df['Seasonality'] == season].copy()
        if len(season_df) == 0:
            monthly_results[season] = None
            monthly_samples[season] = None
            continue
        season_df['YearMonth'] = season_df['Date'].dt.to_period('M')
        months = season_df['YearMonth'].unique()
        fill_sample = []
        for month in months:
            month_df = season_df[season_df['YearMonth'] == month]
            n_days = len(month_df)
            forecast_sum = month_df['Demand Forecast'].sum()
            errors = (month_df['Units Sold'] - month_df['Demand Forecast']).values
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
            st.warning("There is a significant difference in service levels between seasons (reject H₀).")
        else:
            st.success("No significant difference in service levels between seasons (fail to reject H₀).")
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

    # Calculate minimum safety stock
    st.subheader("Recommended Minimum Safety Stock per Season (Monthly Aggregation)")
    min_safety_stock = {}
    for season in season_names:
        season_df = filtered_df[filtered_df['Seasonality'] == season].copy()
        if len(season_df) == 0:
            min_safety_stock[season] = None
            continue
        season_df['YearMonth'] = season_df['Date'].dt.to_period('M')
        months = season_df['YearMonth'].unique()
        found = False
        for ss in range(0, 51):
            fill_sample = []
            for month in months:
                month_df = season_df[season_df['YearMonth'] == month]
                n_days = len(month_df)
                forecast_sum = month_df['Demand Forecast'].sum()
                errors = (month_df['Units Sold'] - month_df['Demand Forecast']).values
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

if __name__ == "__main__":
    main()
