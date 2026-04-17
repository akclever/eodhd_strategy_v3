from eodhd_strategy.fmp_mapper import map_beneish_components
import pandas as pd

# Create test dataframes with 2 periods
income = pd.DataFrame({
    'symbol': ['AAPL', 'AAPL'], 
    'date': ['2022-01-01', '2023-01-01'], 
    'revenue': [90, 100], 
    'net_income': [45, 50],
    'gross_profit': [60, 70],
    'sga_expenses': [15, 18]  # Added for SGAI
})
balance = pd.DataFrame({
    'symbol': ['AAPL', 'AAPL'], 
    'date': ['2022-01-01', '2023-01-01'], 
    'total_assets': [180, 200], 
    'net_receivables': [25, 30],
    'total_liabilities': [80, 90],
    'cash_and_equivalents': [20, 25],  # Added for AQI
    'inventory': [10, 12]  # Added for AQI
})
cashflow = pd.DataFrame({
    'symbol': ['AAPL', 'AAPL'], 
    'date': ['2022-01-01', '2023-01-01'], 
    'operating_cash_flow': [35, 40],
    'capital_expenditure': [10, 12]  # Added for DEPI
})

print('Calling map_beneish_components with 2 periods...')
result = map_beneish_components(income, balance, cashflow)
print(f'Result: {len(result)} rows')
if not result.empty:
    print(f'beneish_m_score: {result["beneish_m_score"].iloc[0]}')
    print(f'All columns: {result.columns.tolist()}')
    # Print all component values
    for col in ['dsri', 'gmi', 'aqi', 'sgi', 'depi', 'sgai', 'lvgi', 'tata']:
        print(f'  {col}: {result[col].iloc[0]}')
