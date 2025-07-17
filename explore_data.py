import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Load the CSV file
print("Loading dataset...")
df = pd.read_csv('crunchbase.csv')

# Explore the dataset structure
print(f'Dataset shape: {df.shape}')
print('\nFirst few column names:')
print(df.columns.tolist()[:20])

# Look for investor-related columns
investor_columns = [col for col in df.columns if 'investor' in col.lower()]
print(f'\nInvestor-related columns: {investor_columns}')

# Check if there's an 'investors' column
if 'investors' in df.columns:
    print('\nSample investors data:')
    sample_investors = df['investors'].dropna().head()
    for i, investor_data in enumerate(sample_investors):
        print(f"Row {i}: {investor_data}")
    print(f'\nInvestors column data type: {df["investors"].dtype}')
    print(f'Non-null values in investors column: {df["investors"].notna().sum()}')
else:
    print('\nNo "investors" column found.')
    
# Check other potential funding-related columns
funding_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ['funding', 'round', 'investment'])]
print(f'\nFunding-related columns: {funding_columns}') 