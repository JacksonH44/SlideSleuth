'''
  A program that takes in the 4 scorers excel files and performs a label smoothing process
  in which the combination of each case is given a continuous value in [0, 1]

  Author: Jackson Howe
  Date Created: June 6, 2023
  Last Updated: June 6, 2023
'''

import numpy as np
import pandas as pd

'''
  A function that takes in a dataframe representing a case and computes the smoothed label 
  value.
'''

def smooth_label(df):
  totals = df.sum(axis=1)
  totals = totals / 400
  label = (totals.loc['invasive'] * 1) + (totals.loc['probable invasive'] * 0.5) + (totals.loc['probable noninvasive'] * 0.5)
  return label

if __name__ == '__main__':
  # Read in the 4 physicians scoring excel files

  tsao_df = pd.read_excel(
    '../inputs/raw/CK7 study_database_rescoring_final_TSAOv2.xlsx', 
    sheet_name='CK7',
    names=['case', 'invasive', 'probable invasive', 'probable noninvasive', 'noninvasive', 'simple', 'complex', 'single cell', 'comments'],
    usecols='B:E'
    )
  tsao_df = tsao_df.fillna(0)

  ey_df = pd.read_excel(
      '../inputs/raw/CK7 study_database_rescoring_final_EY.xlsx',
      sheet_name='CK7',
      names=['case', 'invasive', 'probable invasive', 'probable noninvasive',
             'noninvasive', 'simple', 'complex', 'single cell', 'comments'],
      usecols='B:E'
  )
  ey_df = ey_df.fillna(0)

  mrc_df = pd.read_excel(
      '../inputs/raw/CK7 study_database_rescoring_final_MRCv2.xlsx',
      sheet_name='CK7',
      names=['case', 'invasive', 'probable invasive', 'probable noninvasive',
             'noninvasive', 'simple', 'complex', 'single cell', 'comments'],
      usecols='B:E'
  )
  mrc_df = mrc_df.fillna(0)

  najd_df = pd.read_excel(
      '../inputs/raw/CK7 study_database_rescoring_final-Najd.xlsx',
      sheet_name='CK7',
      names=['case', 'invasive', 'probable invasive', 'probable noninvasive',
             'noninvasive', 'simple', 'complex', 'single cell', 'comments'],
      usecols='B:E'
  )
  najd_df = najd_df.fillna(0)
  
  label_totals = []

  # Loop through all the rows
  for row in range(tsao_df.shape[0]):
    # Concatenate the row from each dataframe and create a dataframe for each case
    merged = pd.concat([tsao_df.iloc[row], ey_df.iloc[row], mrc_df.iloc[row], najd_df.iloc[row]], axis=1)

    # Generate the smoothed label for the case
    label = smooth_label(merged)

    # Add to master list of labels
    label_totals.append(float(f'{label:.5f}'))

  print(label_totals)