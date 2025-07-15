import pandas as pd
import numpy as np
import os
import glob

# --- Configuration ---
# Define input filenames
mapping_filename = './data/Mapping.xlsx' # Hardcoded mapping file path
main_data_filename = './data/344.csv' # Hardcoded main data file path

# Define output filenames
output_filename = './data/output_data_mapped.csv'
nans_output_filename = './data/output_blank_ids.csv'

# Define chunk size for reading main data file
chunk_size = 125000 # chunk size

# --- Ensure Output Directory Exists ---
output_dir = os.path.dirname(output_filename)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: '{output_dir}'")

# Remove existing output files to ensure a clean run
if os.path.exists(output_filename):
    os.remove(output_filename)
    print(f"Removed existing output file: '{output_filename}'")

if os.path.exists(nans_output_filename):
    os.remove(nans_output_filename)
    print(f"Removed existing unmapped values file: '{nans_output_filename}'")


# --- Step 1: Load and prepare mapping data ---
print("\n--- Step 1: Loading mapping data ---")
# Load the mapping data directly from the hardcoded path
try:
    df_mapping = pd.read_excel(mapping_filename)
    print("Mapping file loaded successfully.")

    # Clean up column names for pandas processing
    df_mapping.columns = df_mapping.columns.str.replace('[^0-9a-zA-Z_]+', '_', regex=True)
    df_mapping.columns = df_mapping.columns.str.replace('__', '_', regex=False)
    df_mapping.columns = df_mapping.columns.str.strip('_')

    # Dynamically find the correct column names based on likely content
    mapping_cols = df_mapping.columns.tolist()
    amt_senf_col_mapping = next((col for col in mapping_cols if col.lower().replace('_','') == 'amtsenf' or 'amtsenf' in col.lower()), None)
    cat_no_col_mapping = next((col for col in mapping_cols if col.lower().replace('_','') == 'catno' or 'cat_no' in col.lower()), None)
    avg_value_col_mapping = next((col for col in mapping_cols if 'avg' in col.lower()), None)


    if not all([amt_senf_col_mapping , cat_no_col_mapping, avg_value_col_mapping]):
         raise ValueError(f"Could not automatically determine required column names in mapping table '{mapping_filename}'. Found: {mapping_cols}")

    # Create the composite key in the mapping DataFrame
    # Use .fillna('') before astype(str) to handle potential NaNs in key columns
    df_mapping['composite_key'] = df_mapping[amt_senf_col_mapping].fillna('').astype(str) + '_' + df_mapping[cat_no_col_mapping].fillna('').astype(str)

    # Select only the necessary columns for the lookup structure
    df_mapping_lookup = df_mapping[['composite_key', avg_value_col_mapping]].copy()

    # Rename the average value column for consistency in the lookup
    df_mapping_lookup.rename(columns={avg_value_col_mapping: 'avg_value'}, inplace=True)

    # Set composite_key as the index for faster lookups
    df_mapping_lookup.set_index('composite_key', inplace=True)

    print("Mapping data indexed.")
    #print(f"Mapping lookup structure has {len(df_mapping_lookup)} entries.")

except FileNotFoundError:
    print(f"Error: Mapping file not found at '{mapping_filename}'")
    raise # Re-raise the exception to stop execution
except Exception as e:
    print(f"An error occurred during mapping data preparation: {e}")
    raise # Re-raise the exception


# --- Step 2 & 3 & 4 & 5: Process main data in chunks, Map values, Collect unmapped, Write processed chunks ---
print("\n---  Step 2: Loading main data ---")

# Initialize flag for writing unmapped data header
is_first_unmapped_chunk = True

# Load the main data in chunks directly from the hardcoded path
try:
    # Reading as string is safer for avoiding mixed type issues with chunks.
    csv_reader = pd.read_csv(main_data_filename, chunksize=chunk_size, dtype=str)
    print("Reader created successfully.")

    is_first_chunk = True # Flag to handle processed data header writing

    # Attempt to get column names including potential AMT columns before processing chunks
    try:
        header_df = pd.read_csv(main_data_filename, nrows=0, dtype=str)
        all_data_cols = header_df.columns.tolist()
        # Clean column names for consistency with mapping
        cleaned_cols = pd.Series(all_data_cols).str.replace('[^0-9a-zA-Z_]+', '_', regex=True).str.replace('__', '_', regex=False).str.strip('_').tolist()
        all_data_cols = cleaned_cols # Update the list with cleaned names
        amt_columns_header = [col for col in all_data_cols if col.lower().startswith('amt_')]
        # Identify the ID column, cleaning its name for consistency
        id_column_data = 'id' if 'id' in all_data_cols else ('head_id' if 'head_id' in all_data_cols else all_data_cols[0]) # Attempt to find an ID column
        # Apply cleaning to the identified ID column name as well
        id_column_data = pd.Series([id_column_data]).str.replace('[^0-9a-zA-Z_]+', '_', regex=True).str.replace('__', '_', regex=False).str.strip('_').tolist()[0]


        if not amt_columns_header:
             print("Warning: No columns starting with 'AMT_' found in the main data file header. No mapping will be performed.")

        if id_column_data not in all_data_cols:
             print(f"Warning: Identified ID column '{id_column_data}' not found in header columns.")

    except FileNotFoundError:
        # This specific FileNotFoundError should be caught by the outer try block
        # Re-raise to be caught by the outer block
        raise
    except Exception as e:
        print(f"Error reading header or identifying columns: {e}")
        raise # Re-raise any other exceptions

	
    print("\n---  Step 3: Processing main data  ---")
    for i, chunk_df in enumerate(csv_reader):
        print(f"Processing batch {i+1}...")

        # Clean column names for the chunk as well, for consistency
        chunk_df.columns = chunk_df.columns.str.replace('[^0-9a-zA-Z_]+', '_', regex=True).str.replace('__', '_', regex=False).str.strip('_')

        # Ensure ID column exists in this chunk before processing
        if id_column_data not in chunk_df.columns:
             print(f"Error: ID column '{id_column_data}' not found in chunk {i+1}. Cannot process chunk.")
             continue # Skip this chunk

        # Identify AMT columns in the current chunk based on the header identification
        # This ensures consistency across chunks even if a chunk is empty for an AMT col
        amt_columns_chunk = [col for col in amt_columns_header if col in chunk_df.columns]

        # --- Collect IDs of rows with unmapped data for the current chunk ---
        # This list will store the IDs of rows that have at least one unmapped value in any AMT column in this chunk
        unmapped_row_ids_in_chunk = set()


        # --- Process Each AMT Column within the Chunk ---
        for col_name in amt_columns_chunk:
            # Select only the ID and the current AMT column for processing
            temp_df_col = chunk_df[[id_column_data, col_name]].copy()

            # --- Identify initially blank values in the current AMT column and collect their IDs ---
            # Rows where the original value in this AMT column is NaN or empty string
            blank_entries = temp_df_col[temp_df_col[col_name].isna() | (temp_df_col[col_name] == '')].copy()

            # If there are blank entries, add their IDs to the set of unmapped row IDs
            if not blank_entries.empty:
                 unmapped_row_ids_in_chunk.update(blank_entries[id_column_data].tolist())
            # --- End Identify initially blank values ---


            # Filter out rows where the original value in this AMT column is NaN or empty string for mapping lookup
            temp_df_col_clean = temp_df_col.dropna(subset=[col_name]).copy()
            temp_df_col_clean = temp_df_col_clean[temp_df_col_clean[col_name] != ''].copy()


            if not temp_df_col_clean.empty:
                # Create composite key for this specific column's values
                # Use .fillna('') before astype(str) to handle potential NaNs in the original value column
                temp_df_col_clean['composite_key'] = col_name + '_' + temp_df_col_clean[col_name].fillna('').astype(str)

                # Perform the lookup/merge with the mapping data
                # Use a left merge to keep all rows from the cleaned chunk data
                # The mapped_value column will be NaN for unmapped entries
                # Ensure df_mapping_lookup is available from a previous step
                if 'df_mapping_lookup' not in locals():
                    print("Error: Mapping lookup data (df_mapping_lookup) not found. Cannot perform mapping.")
                    # Depending on severity, you might want to exit or skip mapping for this chunk/column
                    continue # Skip mapping for this column


                # Perform the merge with the mapping lookup
                # The merged_df will have the original data columns + 'composite_key' + 'avg_value' (or NaN)
                merged_df = pd.merge(
                    temp_df_col_clean,
                    df_mapping_lookup,
                    left_on='composite_key',
                    right_index=True, # Merge with the index of df_mapping_lookup
                    how='left'
                )

                # --- Identify entries with failed lookups and collect their row IDs ---
                # Unmapped entries are those where the merge resulted in a NaN 'avg_value'
                unmapped_entries_lookup_fail = merged_df[merged_df['avg_value'].isna()].copy()
                if not unmapped_entries_lookup_fail.empty:
                     # Add the IDs of rows with failed lookups to the set of unmapped row IDs
                     unmapped_row_ids_in_chunk.update(unmapped_entries_lookup_fail[id_column_data].tolist())
                # --- End Identify entries with failed lookups ---


                # Update the original chunk DataFrame with the mapped values
                # Create a Series for the update, aligned to the original chunk index
                update_series_nan = pd.Series(index=chunk_df.index, dtype=object)
                # For the rows that were processed (in temp_df_col_clean), assign the mapped value or np.nan
                # Aligning by index explicitly here ensures correct row correspondence
                mapped_values_for_update = merged_df['avg_value'].values # Get values from merged_df
                original_indices_for_update = temp_df_col_clean.index # Get the original indices from the chunk
                update_series_nan.loc[original_indices_for_update] = mapped_values_for_update

                # Update the chunk_df column using .loc and the original index
                # Where update_series_nan is not NaN, replace the value in chunk_df
                # Where update_series_nan IS NaN, the value in chunk_df will become NaN if it wasn't already (for processed rows), or stay NaN (for filtered out rows).
                chunk_df.loc[temp_df_col_clean.index, col_name] = update_series_nan.loc[original_indices_for_update]


            # If temp_df_col_clean was empty, no processing or updating is needed for this column in this chunk.
            # The original values (which were all NaN/empty) remain in chunk_df.

        # --- Write collected unmapped row IDs for the current chunk to output (Chunk-by-Chunk Write) ---
        if unmapped_row_ids_in_chunk:
            # Convert the set of unique unmapped row IDs to a DataFrame
            chunk_unmapped_df = pd.DataFrame({id_column_data: list(unmapped_row_ids_in_chunk)})

            # Ensure the ID column name is consistent in the unmapped output
            nans_header_id_col = id_column_data if 'id_column_data' in locals() and id_column_data is not None else 'ID'
            if nans_header_id_col in chunk_unmapped_df.columns and chunk_unmapped_df.columns[0] != nans_header_id_col:
                 cols = [nans_header_id_col] + [col for col in chunk_unmapped_df.columns if col != nans_header_id_col]
                 chunk_unmapped_df = chunk_unmapped_df[cols]

            # Write to the unmapped values file
            if is_first_unmapped_chunk:
                # Write with header
                chunk_unmapped_df.to_csv(nans_output_filename, index=False, mode='w', encoding='utf-8')
                is_first_unmapped_chunk = False
            else:
                # Append without header
                chunk_unmapped_df.to_csv(nans_output_filename, index=False, mode='a', header=False, encoding='utf-8')
            #print(f"Wrote {len(unmapped_row_ids_in_chunk)} unique unmapped row IDs from chunk {i+1} to '{nans_output_filename}'.")


        # --- Write the processed chunk to the output file ---
        # Ensure the original ID column name is used for the output file
        if id_column_data in chunk_df.columns and chunk_df.columns[0] != id_column_data:
             # Reorder columns to put the ID column first if it's not already
             cols = [id_column_data] + [col for col in chunk_df.columns if col != id_column_data]
             chunk_df = chunk_df[cols]


        if is_first_chunk:
            # Write the header for the first chunk
            chunk_df.to_csv(output_filename, index=False, mode='w', encoding='utf-8')
            is_first_chunk = False
        else:
            # Append subsequent chunks without the header
            chunk_df.to_csv(output_filename, index=False, mode='a', header=False, encoding='utf-8')

        #print(f"Finished processing chunk {i+1}.")


    print(f"\n--- Finished Processing main data. Mapped data copied into {output_filename} ---")


except FileNotFoundError:
     print(f"Error: Main data file not found at '{main_data_filename}'")
     # Re-raise the exception to stop execution
     raise
except Exception as e:
    print(f"An error occurred during chunk processing: {e}")
    # Re-raise the exception to stop execution
    raise

# --- Final check/creation of empty unmapped values file if no unmapped data was ever found ---
# This handles the case where no chunks contained unmapped data, but we still need the file created with headers.
# This check should now be done outside the loop, but only if is_first_unmapped_chunk is still True.
if is_first_unmapped_chunk:
    print(f"\n--- Step 6: No unmapped values found in any chunk. Creating empty unmapped values file: '{nans_output_filename}' ---")
    # Create an empty nans.csv file with just headers if no unmapped values
    # Use the identified id_column_data if available, otherwise a default
    nans_header_id_col = id_column_data if 'id_column_data' in locals() and id_column_data is not None else 'ID'
    # Create an empty DataFrame with just the ID column header
    empty_nans_df = pd.DataFrame(columns=[nans_header_id_col])
    empty_nans_df.to_csv(nans_output_filename, index=False, mode='w', encoding='utf-8')
    print(f"Created empty unmapped values file: '{nans_output_filename}'")
else:
     print(f"\n--- Finished processing blank inputs. Blank ids copied into '{nans_output_filename}' ---")
     # The unmapped data was written chunk by chunk, so no final concatenation needed.


print("\nJob finished.")