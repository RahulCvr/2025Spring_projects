# Inventory Policy Simulator 

A Streamlit-based application for simulating and analyzing inventory management policies, with a focus on comparing fixed vs. dynamic reorder policies and analyzing seasonal effects on service levels.

## Overview

This project provides a comprehensive tool for inventory management analysis, featuring:

1. **Dynamic vs. Fixed Policy Comparison**: Simulates and compares the performance of dynamic and fixed reorder point policies
2. **Seasonal Analysis**: Analyzes the impact of seasonal variations on service levels
3. **Safety Stock Optimization**: Recommends optimal safety stock levels for different seasons

## Experiments
### Experiment 1

**H₀:** There is no difference in average total cost between the Fixed and Dynamic policies.  
**H₁:** The Dynamic policy has a lower average total cost than the Fixed policy.

### Experiment 2

**H₀:** A single safety stock level delivers consistent service levels across all seasons.  
**H₁:** Seasonal variability leads to significant differences in service levels, requiring tailored safety-stock levels.

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
cd 2025Spring_projects
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
   - If you'd like to experiment with sample data - utilize the sample datasets provided in the repo

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

Literature Review
- https://www.sciencedirect.com/science/article/abs/pii/S0925527308001217
- https://arxiv.org/abs/2310.01079

