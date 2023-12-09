import sys  # For command line file input.
import pandas as pd  # For easy date frame manipulation
from sklearn.model_selection import train_test_split  # To split processed data

# Function to create dummy variables for a specific column in a DataFrame


def dummy_variable_col(df, col):
    # Create dummy variables for the specified column
    df_temp = pd.get_dummies(df[col])
    df_temp = df_temp.astype(int)
    # Concatenate the dummy variables to the original DataFrame and reindex
    df = pd.concat([df, df_temp], axis=1,).reindex(df.index)
    # Drop the original column as dummy variables are created
    df.drop(col, axis=1, inplace=True)
    return df


# Check if the correct number of command-line arguments is provided
if len(sys.argv) == 3:
    # Assign input and output file names from command-line arguments
    file_input = sys.argv[1]
    file_output = sys.argv[2]
else:
    # Print an error message if the number of arguments is invalid
    print("Error: Invalid number of arguments.", file=sys.stderr)

# Read the input CSV file into a Pandas DataFrame
df = pd.read_csv(file_input)
df = df.drop(['id', 'color'], axis=1)
df = df.dropna()  # Drop any null (empty) values
df = dummy_variable_col(df, 'habitat')
df['Gender'] = df['Gender'].replace({True: 'male', False: 'female'})

# Split the DataFrame into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

# Write the training and testing sets to CSV files
train_df.to_csv(file_output + '_train.csv', index=False)
test_df.to_csv(file_output + '_test.csv', index=False)
