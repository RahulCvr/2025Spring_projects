# Inventory Policy Simulator

A Streamlit-based application for simulating and analyzing inventory management policies, with a focus on comparing fixed vs. dynamic reorder policies and analyzing seasonal effects on service levels.

## Overview

This project provides a comprehensive tool for inventory management analysis, featuring:

1. **Dynamic vs. Fixed Policy Comparison**: Simulates and compares the performance of dynamic and fixed reorder point policies
2. **Seasonal Analysis**: Analyzes the impact of seasonal variations on service levels
3. **Safety Stock Optimization**: Recommends optimal safety stock levels for different seasons
4. **Interactive UI**: User-friendly interface for parameter adjustment and result visualization

## Features

- **Policy Comparison**: Compare fixed and dynamic inventory policies
- **Cost Analysis**: Analyze holding costs, stockout costs, and ordering costs
- **Statistical Testing**: Perform t-tests and ANOVA analysis
- **Seasonal Analysis**: Evaluate service levels across different seasons
- **Safety Stock Recommendations**: Calculate optimal safety stock levels
- **Interactive Visualizations**: View cost distributions and seasonal trends
- **Configurable Parameters**: Adjust simulation parameters through a user-friendly interface

## Prerequisites

- Python 3.7+
- Required Python packages (install using `pip install -r requirements.txt`):
  - streamlit
  - numpy
  - pandas
  - scikit-learn
  - scipy
  - matplotlib
  - seaborn
  - PyYAML

## Installation

1. Clone the repository:
```bash
git clone https://github.com/RahulCvr/2025Spring_projects
cd inventory-policy-simulator
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Create a `config.yaml` file in the project directory with the following structure or use the default one:
```yaml
columns:
  date: "Date"
  store_id: "Store ID"
  product_id: "Product ID"
  demand_forecast: "Demand Forecast"
  inventory_level: "Inventory Level"
  units_ordered: "Units Ordered"
  units_sold: "Units Sold"
  category: "Category"
  region: "Region"
  weather_condition: "Weather Condition"
  seasonality: "Seasonality"
  holiday_promotion: "Holiday/Promotion"

simulation:
  service_level_range:
    min: 80
    max: 99
    default: 95
  holding_cost_default: 0.05
  stockout_cost_default: 2.00
  ordering_cost_default: 50.00
  reorder_point_range:
    min: 50
    max: 300
    default: 150
    step: 10
  fixed_order_quantity_default: 300
  monthly_safety_stock_default: 20
  monthly_target_fill_range:
    min: 0.80
    max: 0.99
    default: 0.95

ui:
  title: "Inventory Policy Simulator: Fixed vs Dynamic - Weekly Ordering"
  sidebar_header: "Simulation Settings"
```

## Usage

1. Prepare your data:
   - Create a CSV file with the required columns as specified in the config file
   - Ensure your data includes all necessary fields (date, product ID, demand forecast, etc.)

2. Run the application:
```bash
streamlit run main.py
```

3. Using the application:
   - Upload your CSV file through the file uploader
   - Select a product from the dropdown menu
   - Adjust simulation parameters in the sidebar:
     - Service Level
     - Holding Cost
     - Stockout Cost
     - Ordering Cost
     - Fixed Reorder Point
     - Manual Order Quantity
     - Safety Stock
     - Target Fill Rate

4. View results:
   - Simulation results comparing fixed and dynamic policies
   - Cost distribution visualizations
   - Statistical analysis results
   - Seasonal analysis
   - Safety stock recommendations

## Input Data Requirements

Your CSV file should contain the following columns (names can be configured in config.yaml):
- Date: Timestamp of the record
- Product ID: Unique identifier for products
- Demand Forecast: Predicted demand
- Inventory Level: Current inventory level
- Units Ordered: Number of units ordered
- Units Sold: Number of units sold
- Category: Product category
- Region: Geographic region
- Weather Condition: Weather status
- Seasonality: Seasonal indicator
- Holiday/Promotion: Special event indicator


## Acknowledgments

- Streamlit for the web application framework
- Scikit-learn for machine learning capabilities
- Pandas and NumPy for data manipulation
- Matplotlib and Seaborn for visualization 
