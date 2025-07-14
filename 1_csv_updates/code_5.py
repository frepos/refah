import sqlite3
import pandas as pd
import os
import glob
import numpy as np

# --- Configuration ---
db_filename = './data/optimized_db.sqlite' # Database file for this strategy
output_filename = './data/output_data_mapped.csv' # Output file for the mapped data
nans_output_filename = './data/output_blanks.csv' # Output file for unmapped data

# --- Input File Names (can be auto-detected later if needed) ---
main_data_filename = './data/344.csv' # Assuming head.csv is in the default Colab directory
mapping_filename = './data/Mapping.xlsx' # Assuming Mapping.xlsx is in the default Colab directory

# --- Database Table Names ---
tn_main_data = 'main_data_table'
tn_mapping_mid = 'mapping_midtable'
tn_unmapped_temp = 'temp_unmapped_values' # Temporary table to store unmapped original values

# Define chunk size for reading large CSV files during initial import
import_chunk_size = 100000 # Example chunk size for import

# --- Ensure Database is Clean ---
if os.path.exists(db_filename):
    os.remove(db_filename)
    print(f"Removed existing database file: '{db_filename}' for a clean run.")

# --- Connect to Database ---
conn = None # Initialize connection to None
cursor = None # Initialize cursor to None
try:
    conn = sqlite3.connect(db_filename)
    cursor = conn.cursor()
    print(f"Successfully connected to SQLite database: {db_filename}")
except Exception as e:
    print(f"Error connecting to database: {e}")
    # If connection fails, we cannot proceed.
    raise

# --- Step 1: Load Mapping File and Create mapping_midtable (with Composite Key and Index) ---
try:
    print(f"\n--- Step 1: Processing mapping file into '{tn_mapping_mid}' ---")
    # Read the Mapping.xlsx file into a pandas DataFrame
    df_mapping = pd.read_excel(mapping_filename)
    print("Mapping file loaded successfully into pandas DataFrame.")

    # Clean up column names for pandas processing
    df_mapping.columns = df_mapping.columns.str.replace('[^0-9a-zA-Z_]+', '_', regex=True)
    df_mapping.columns = df_mapping.columns.str.replace('__', '_', regex=False)
    df_mapping.columns = df_mapping.columns.str.strip('_')

    # Dynamically find the correct column names based on likely content
    mapping_cols = df_mapping.columns.tolist()
    amt_senf_col_mapping = next((col for col in mapping_cols if col.lower().replace('_','') == 'amtsenf' or 'amtsenf' in col.lower()), None)
    cat_no_col_mapping = next((col for col in mapping_cols if col.lower().replace('_','') == 'catno' or 'cat_no' in col.lower()), None)
    avg_value_col_mapping = next((col for col in mapping_cols if 'avg' in col.lower()), None)

    if not all([amt_senf_col_mapping, cat_no_col_mapping, avg_value_col_mapping]):
         raise ValueError(f"Could not automatically determine required column names in mapping table '{mapping_filename}'. Found: {mapping_cols}")

    # Create the composite key in the DataFrame
    # Handle potential NaN values in the key columns by converting to string
    df_mapping['composite_key'] = df_mapping[amt_senf_col_mapping].fillna('').astype(str) + '_' + df_mapping[cat_no_col_mapping].fillna('').astype(str)

    # Select only the necessary columns for the intermediate table
    df_mapping_mid = df_mapping[['composite_key', avg_value_col_mapping]].copy()
    # Rename the average value column to a consistent name for the database table
    df_mapping_mid.rename(columns={avg_value_col_mapping: 'avg_value'}, inplace=True)

    # Import data directly into mapping_midtable
    print(f"\nImporting data into intermediate mapping table '{tn_mapping_mid}'...")
    # if_exists='replace' will drop the table and recreate it if it exists
    df_mapping_mid.to_sql(tn_mapping_mid, conn, index=False, if_exists='replace')
    print(f"Data successfully imported into table '{tn_mapping_mid}'.")

    # Add index to composite_key in mapping_midtable
    index_name = f'idx_{tn_mapping_mid}_composite_key'
    print(f"Creating index '{index_name}' on '{tn_mapping_mid}'...")
    cursor.execute(f"CREATE INDEX IF NOT EXISTS \"{index_name}\" ON \"{tn_mapping_mid}\" (composite_key);")
    print(f"Index '{index_name}' created.")

    conn.commit() # Commit changes
    print(f"Intermediate mapping table '{tn_mapping_mid}' created, populated, and indexed successfully.")

except Exception as e:
    print(f"An error occurred during mapping file processing: {e}")
    if conn:
        conn.rollback() # Roll back changes if an error occurred
    # Close connection before raising error
    if conn:
        conn.close()
    raise # Re-raise the exception


# --- Step 2: Load Main Data File into main_data_table (Chunked) ---
try:
    print(f"\n--- Step 2: Loading main data file into '{tn_main_data}' (Chunked) ---")
    # Drop the main_data_table if it exists to ensure a clean start
    cursor.execute(f"DROP TABLE IF EXISTS \"{tn_main_data}\"")
    conn.commit()
    print(f"Dropped existing table '{tn_main_data}' if it existed.")

    # Read the main data file in chunks and append to the database table
    csv_reader = pd.read_csv(main_data_filename, chunksize=import_chunk_size, dtype=str)

    total_rows_imported = 0
    is_first_chunk = True

    print(f"Starting chunked import of '{main_data_filename}' into table '{tn_main_data}'...")

    for i, chunk_df in enumerate(csv_reader):
        print(f"Importing chunk {i+1}...")
        # Clean column names for SQL compatibility
        chunk_df.columns = chunk_df.columns.str.replace('[^0-9a-zA-Z_]+', '_', regex=True)
        chunk_df.columns = chunk_df.columns.str.replace('__', '_', regex=False)
        chunk_df.columns = chunk_df.columns.str.strip('_')

        # Use pandas to_sql to append chunks. SQLite infers schema from the first chunk.
        # if_exists='append' is used after the initial drop
        chunk_df.to_sql(tn_main_data, conn, index=False, if_exists='append')
        total_rows_imported += len(chunk_df)
        print(f"Finished importing chunk {i+1}. Total rows imported: {total_rows_imported}")

    conn.commit() # Commit after all chunks are imported
    print(f"\nData from '{main_data_filename}' successfully imported into table '{tn_main_data}' in chunks.")

except FileNotFoundError:
     print(f"Error: Main data file not found at '{main_data_filename}'")
     if conn:
         conn.close()
     raise # Re-raise the exception
except Exception as e:
    print(f"An error occurred during chunked import of main data file: {e}")
    if conn:
        conn.rollback() # Roll back changes if an error occurred
        conn.close()
    raise # Re-raise the exception


# --- Step 3: Identify AMT Columns in the Main Data Table ---
try:
    print(f"\n--- Step 3: Identifying AMT columns in '{tn_main_data}' ---")
    # Get column names from the database table schema
    cursor.execute(f"PRAGMA table_info(\"{tn_main_data}\");")
    main_data_cols_info = cursor.fetchall()
    main_data_cols = [col[1] for col in main_data_cols_info]

    amt_columns = [col for col in main_data_cols if col.lower().startswith('amt_')]
    if not amt_columns:
        print("Warning: No AMT columns found in the main data table. No mapping will be performed.")
    else:
        print(f"Found {len(amt_columns)} AMT columns: {amt_columns}")

    # Identify the ID column for later export and unmapped tracking
    id_column_data = 'id' if 'id' in main_data_cols else ('head_id' if 'head_id' in main_data_cols else main_data_cols[0])
    print(f"Identified ID column: '{id_column_data}'")

except Exception as e:
    print(f"An error occurred while identifying AMT columns: {e}")
    if conn:
        conn.close()
    raise # Re-raise the exception


# --- Step 4: Iteratively Update AMT Columns with Mapped Values in the Database ---
if amt_columns:
    try:
        print(f"\n--- Step 4: Iteratively updating AMT columns in '{tn_main_data}' with mapped values ---")

        # Create temporary table to store unmapped original values before they are overwritten
        cursor.execute(f"DROP TABLE IF EXISTS \"{tn_unmapped_temp}\"")
        create_unmapped_temp_sql = f"""
        CREATE TABLE "{tn_unmapped_temp}" (
            original_id TEXT,
            original_value TEXT,
            column_name TEXT
        );
        """
        cursor.execute(create_unmapped_temp_sql)
        conn.commit()
        print(f"Temporary table '{tn_unmapped_temp}' created for storing unmapped values.")


        for i, col_name in enumerate(amt_columns):
            print(f"Updating column '{col_name}' ({i+1}/{len(amt_columns)})...")

            # --- Store unmapped original values for the current column before updating ---
            # Select original non-NULL values that do NOT have a match in mapping_midtable
            # A value doesn't have a match if the composite key derived from it is not found in mapping_midtable
            insert_unmapped_sql = f"""
            INSERT INTO "{tn_unmapped_temp}" (original_id, original_value, column_name)
            SELECT
                "{id_column_data}",
                "{col_name}",
                '{col_name}'
            FROM "{tn_main_data}" AS t1
            WHERE "{col_name}" IS NOT NULL AND "{col_name}" != '' -- Consider only non-empty original values
              AND NOT EXISTS (
                  SELECT 1
                  FROM "{tn_mapping_mid}" AS t2
                  WHERE ('{col_name}' || '_' || t1."{col_name}") = t2.composite_key
              );
            """
            cursor.execute(insert_unmapped_sql)
            conn.commit()
            print(f"Stored unmapped original values for '{col_name}'.")


            # --- Update the current AMT column with mapped values ---
            # Use a LEFT JOIN to get the mapped value, set to NULL if no match found
            # The WHERE clause limits updates to rows where a composite key can be formed (original value is not null/empty)
            # and where a match is actually found in the mapping table (using an IN or EXISTS subquery is often efficient for updates)

            update_sql = f"""
            UPDATE "{tn_main_data}"
            SET "{col_name}" = (
                SELECT mm.avg_value
                FROM "{tn_mapping_mid}" AS mm
                WHERE ('{col_name}' || '_' || "{col_name}") = mm.composite_key
            )
            WHERE EXISTS (
                SELECT 1
                FROM "{tn_mapping_mid}" AS mm_exists
                WHERE ('{col_name}' || '_' || "{col_name}") = mm_exists.composite_key
            );
            """
            cursor.execute(update_sql)
            conn.commit()
            print(f"Updated '{col_name}' with mapped values.")

        print("\nFinished updating all AMT columns.")

    except Exception as e:
        print(f"An error occurred during iterative column updates: {e}")
        if conn:
            conn.rollback() # Roll back changes if an error occurred
            conn.close()
        raise # Re-raise the exception


# --- Step 5: Export the Final Updated Data Table to CSV ---
try:
    print(f"\n--- Step 5: Exporting final updated data to '{output_filename}' ---")
    # Read the final table into a pandas DataFrame for export
    # This might still be memory intensive if the final table is very large
    df_final_data = pd.read_sql_query(f"SELECT * FROM \"{tn_main_data}\"", conn)

    # Ensure the ID column is named correctly for the output if it was changed during import
    if 'original_id' in df_final_data.columns and id_column_data != 'original_id':
         df_final_data.rename(columns={'original_id': id_column_data}, inplace=True)

    # Export to CSV
    df_final_data.to_csv(output_filename, index=False, encoding='utf-8')
    print(f"Final updated data successfully exported to '{output_filename}'")

except Exception as e:
    print(f"An error occurred during final data export: {e}")
    if conn:
        conn.close()
    raise # Re-raise the exception


# --- Step 6: Export Unmapped Values to CSV ---
try:
    print(f"\n--- Step 6: Exporting unmapped values to '{nans_output_filename}' ---")
    # Read the collected unmapped values from the temporary table
    df_unmapped = pd.read_sql_query(f"SELECT * FROM \"{tn_unmapped_temp}\"", conn)

    if not df_unmapped.empty:
        df_unmapped.to_csv(nans_output_filename, index=False, encoding='utf-8')
        print(f"Unmapped values information saved to '{nans_output_filename}'")
    else:
        print("No unmapped values found.")

except Exception as e:
    print(f"An error occurred during unmapped values export: {e}")
    # Continue and close connection even if unmapped export fails
    pass # Do not re-raise, just print error


# --- Clean up temporary tables ---
try:
    print("\nCleaning up temporary tables...")
    cursor.execute(f"DROP TABLE IF EXISTS \"{tn_unmapped_temp}\"")
    # We might choose to keep the main_data_table and mapping_midtable
    # or drop them depending on whether they are needed after the script runs.
    # For a clean run each time, dropping them is good.
    cursor.execute(f"DROP TABLE IF EXISTS \"{tn_main_data}\"")
    cursor.execute(f"DROP TABLE IF EXISTS \"{tn_mapping_mid}\"")
    conn.commit()
    print("Temporary and main tables dropped.")
except Exception as e:
    print(f"An error occurred during table cleanup: {e}")
    pass # Do not re-raise, just print error


# --- Close Database Connection ---
if conn:
    conn.close()
    print("\nDatabase connection closed.")

print("\nOverall iterative SQL update process finished.")