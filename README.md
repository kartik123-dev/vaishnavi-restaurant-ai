# Vaishnavi Restaurant Management System

A Streamlit project for a Class 12 CBSE Artificial Intelligence (Code 834) practical. It records restaurant sales, tracks ingredients, estimates next-day sales, and calculates ingredient purchase requirements.

## Features

- Dashboard for revenue, units sold, and sales trends
- Sales entry, history, and CSV import/export
- Next-day sales estimates using recent averages; Random Forest is used when enough history is available
- Ingredient inventory with add, edit, delete, and CSV import/export
- Editable menu recipes
- Purchase recommendations based on predicted ingredient use, current stock, incoming stock, and safety stock
- CSV files for simple local data storage

## Run the project

1. Install Python 3.11 or a compatible newer version.
2. Open a terminal in this project folder.
3. Install the packages:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Start the app:

   ```bash
   python -m streamlit run streamlit_app.py
   ```

5. Streamlit will show a local address to open in your browser.

## Data files

The app reads and writes `sales.csv`, `inventory.csv`, `recipes.csv`, and `menu.csv` inside the `data/` folder. Sample data is included. Keep a backup of this folder if you want to preserve entered information.

## Notes about predictions

Predictions are estimates, not guarantees. With little or no sales history, the app shows a limited-history method or an insufficient-data message. Results improve as accurate daily sales are entered.

## Quick checks

To run the included basic function checks, install the requirements and run:

```bash
python smoke_test.py streamlit_app.py
```

Before submitting, review the code, make sure you can explain the main functions, and follow your school's rules for project assistance.
