"""
Python equivalent of expxlsx.m
expxlsx(Tables, filename, sheetNames)
- Tables: list of pandas DataFrames [T1, T2, ...]
- filename: string, Excel file name (e.g. "chapterI.xlsx")
- sheetNames: list of strings for each sheet ["T1", "T2", ...]
The Excel file will be saved in the "excel" folder under the project root.
"""
import os
import pandas as pd
def expxlsx(Tables, filename, sheetNames):
    # --- Step 1: Get project root (2 levels up from shared_functions)
    here = os.path.dirname(os.path.abspath(__file__))  # shared_functions folder
    rootDir = os.path.abspath(os.path.join(here, "..", ".."))
    # --- Step 2: Define excel folder ---
    excelDir = os.path.join(rootDir, "excel")
    os.makedirs(excelDir, exist_ok=True)
    # --- Step 3: Build full path ---
    filepath = os.path.join(excelDir, filename)
    # --- Step 4: Write tables ---
    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        for tbl, sheet in zip(Tables, sheetNames):
            tbl.to_excel(writer, sheet_name=sheet, index=False)
    # --- Step 5: Adjust column widths (like MATLAB AutoFit) ---
    from openpyxl import load_workbook
    wb = load_workbook(filepath)
    for sheet in sheetNames:
        ws = wb[sheet]
        for col in ws.columns:
            max_length = 0
            col_letter = col[0].column_letter  # e.g. 'A'
            for cell in col:
                try:
                    val = str(cell.value)
                    if val:
                        max_length = max(max_length, len(val))
                except:
                    pass
            adjusted_width = max_length + 2  # padding
            ws.column_dimensions[col_letter].width = adjusted_width
    wb.save(filepath)
    wb.close()
