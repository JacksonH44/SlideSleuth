"""A script that finds the binary labels for each WSI case, either lepidic-predominant or acinar-predominant.

Date Created: July 28, 2023
Last Updated: July 28, 2023
"""

__author__ = 'Jackson Howe'

import pandas as pd

RAW_FILE = '../../data/raw/lepidic CK7 study_case list_clinical outcome_v1.xlsx'

def create_dfs(file_path):
  """A function that creates a randomization dataframe that details cases and their serialization number, a lepidic dataframe of all lepidic serialization numbers, and a acinar dataframe of all acinar serialization numbers.

  Args:
      file_path (str): Path to the raw excel file

  Returns:
      tuple<pd.DataFrame>: Randomization, lepidic, and acinar dataframes.
  """
  
  randomization_df = pd.read_excel(
    file_path, 
    sheet_name='Randomization',
    header=None,
    names=['serialization', 'case']
  )
  
  print(randomization_df)
  
  lepidic_start = 2
  lepidic_end = 55
  
  acinar_start = 58
  acinar_end = 111
  
  # Extract all of the serialization numbers from the raw excel file.
  final_cases_df = pd.read_excel(
    file_path,
    sheet_name='Final cases',
    usecols='A',
    header=None,
    names=['serialization']
  )
  
  # Split the serialization numbers into lepidic and acinar serializations.
  lepidic_df = final_cases_df.iloc[lepidic_start:lepidic_end+1]
  acinar_df = final_cases_df.iloc[acinar_start:acinar_end+1] 
  
  return randomization_df, lepidic_df, acinar_df

def make_classification_dataframe(randomization_df, lepidic_df, acinar_df):
  """A function that creates a dataframe consisting of a case number and its lepidic/acinar classification.

  Args:
      randomization_df (pd.DataFrame): Dataframe representing the serialization numbers for each case.
      lepidic_df (pd.DataFrame): Dataframe representing the lepidic cases.
      acinar_df (pd.DataFrame): Dataframe representing the acinar cases.

  Returns:
      pd.DataFrame: The final dataframe of cases and classification.
  """
  
  case_dict = {}
  
  types = ['lepidic', 'acinar']
  for subtype in types:
    df = acinar_df
    if subtype == 'lepidic':
      df = lepidic_df
    # For each row in the subtype dataframe, find the case in the randomization dataframe corresponding to that serialization number and add it to the final dataframe.
    for row in df.iterrows():
      serialization = row[1].values[0]
      case = randomization_df[randomization_df['serialization'] == serialization].values[0][1]
      case_dict[case] = f'{subtype}'
      
  final_df = pd.DataFrame.from_dict(case_dict.items())
  return final_df

if __name__ == '__main__':
  randomization_df, lepidic_df, acinar_df = create_dfs(RAW_FILE)
  final_df = make_classification_dataframe(
    randomization_df, 
    lepidic_df, 
    acinar_df
  )
  print(final_df)