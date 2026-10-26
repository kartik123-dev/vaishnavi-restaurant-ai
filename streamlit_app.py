from pathlib import Path
from datetime import date, timedelta
import math
import pandas as pd
import streamlit as st
import plotly.express as px

try:
    from sklearn.ensemble import RandomForestRegressor
except ImportError:
    RandomForestRegressor = None

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

SALES_FILE = DATA_DIR / "sales.csv"
INVENTORY_FILE = DATA_DIR / "inventory.csv"
RECIPES_FILE = DATA_DIR / "recipes.csv"
MENU_FILE = DATA_DIR / "menu.csv"

SALES_COLS = ["date", "item", "quantity", "price", "revenue"]
INVENTORY_COLS = ["ingredient", "stock_quantity", "unit", "minimum_stock", "purchase_cost", "incoming_stock", "safety_stock"]
RECIPE_COLS = ["menu_item", "ingredient", "quantity_per_item"]
MENU_COLS = ["item"]

SAMPLE_MENU = pd.DataFrame({"item": ["Veg Pizza", "Paneer Sandwich", "Masala Tea"]})
SAMPLE_INVENTORY = pd.DataFrame([
    {"ingredient": "Flour", "stock_quantity": 3000.0, "unit": "g", "minimum_stock": 500.0, "purchase_cost": 0.04, "incoming_stock": 0.0, "safety_stock": 300.0},
    {"ingredient": "Cheese", "stock_quantity": 800.0, "unit": "g", "minimum_stock": 200.0, "purchase_cost": 0.35, "incoming_stock": 0.0, "safety_stock": 150.0},
    {"ingredient": "Paneer", "stock_quantity": 700.0, "unit": "g", "minimum_stock": 200.0, "purchase_cost": 0.45, "incoming_stock": 0.0, "safety_stock": 150.0},
    {"ingredient": "Bread", "stock_quantity": 20.0, "unit": "slices", "minimum_stock": 6.0, "purchase_cost": 2.0, "incoming_stock": 0.0, "safety_stock": 4.0},
    {"ingredient": "Tea", "stock_quantity": 500.0, "unit": "g", "minimum_stock": 100.0, "purchase_cost": 0.12, "incoming_stock": 0.0, "safety_stock": 100.0},
    {"ingredient": "Milk", "stock_quantity": 2000.0, "unit": "ml", "minimum_stock": 500.0, "purchase_cost": 0.08, "incoming_stock": 0.0, "safety_stock": 300.0},
])
SAMPLE_RECIPES = pd.DataFrame([
    {"menu_item": "Veg Pizza", "ingredient": "Flour", "quantity_per_item": 200.0},
    {"menu_item": "Veg Pizza", "ingredient": "Cheese", "quantity_per_item": 100.0},
    {"menu_item": "Paneer Sandwich", "ingredient": "Bread", "quantity_per_item": 2.0},
    {"menu_item": "Paneer Sandwich", "ingredient": "Paneer", "quantity_per_item": 80.0},
    {"menu_item": "Masala Tea", "ingredient": "Tea", "quantity_per_item": 5.0},
    {"menu_item": "Masala Tea", "ingredient": "Milk", "quantity_per_item": 100.0},
])

def ensure_files():
    defaults = [(MENU_FILE, SAMPLE_MENU), (INVENTORY_FILE, SAMPLE_INVENTORY),
                (RECIPES_FILE, SAMPLE_RECIPES),
                (SALES_FILE, pd.DataFrame(columns=SALES_COLS))]
    for path, frame in defaults:
        if not path.exists():
            frame.to_csv(path, index=False)

def read_csv(path, columns):
    try:
        frame = pd.read_csv(path)
        for col in columns:
            if col not in frame.columns:
                frame[col] = 0 if col in ("quantity", "price", "revenue", "stock_quantity", "minimum_stock", "purchase_cost", "incoming_stock", "safety_stock", "quantity_per_item") else ""
        return frame[columns]
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
        return pd.DataFrame(columns=columns)

def clean_sales(frame):
    required = ["date", "item", "quantity", "price"]
    if not set(required).issubset(frame.columns):
        raise ValueError("Sales CSV must contain: date, item, quantity, price. Revenue is calculated automatically.")
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    out["item"] = out["item"].astype(str).str.strip()
    out["quantity"] = pd.to_numeric(out["quantity"], errors="coerce")
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    if out["date"].isna().any() or (out["item"] == "").any():
        raise ValueError("Every row needs a valid date and menu item.")
    if out["quantity"].isna().any() or out["price"].isna().any() or (out["quantity"] <= 0).any() or (out["price"] < 0).any():
        raise ValueError("Quantity must be positive and price cannot be negative.")
    out["revenue"] = out["quantity"] * out["price"]
    return out[SALES_COLS]

def clean_inventory(frame):
    required = INVENTORY_COLS
    if not set(required).issubset(frame.columns):
        raise ValueError("Inventory CSV must contain these columns: " + ", ".join(required))
    out = frame[required].copy()
    out["ingredient"] = out["ingredient"].astype(str).str.strip()
    out["unit"] = out["unit"].astype(str).str.strip()
    if (out["ingredient"] == "").any() or (out["unit"] == "").any():
        raise ValueError("Ingredient names and units cannot be blank.")
    for col in ["stock_quantity", "minimum_stock", "purchase_cost", "incoming_stock", "safety_stock"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        if out[col].isna().any() or (out[col] < 0).any():
            raise ValueError(f"{col} must contain non-negative numbers.")
    if out["ingredient"].duplicated().any():
        raise ValueError("Ingredient names must be unique.")
    return out

def save_frame(frame, path):
    frame.to_csv(path, index=False)

def next_day_predictions(sales, menu_items, target_date=None):
    """Return one next-day estimate per menu item, with a transparent method label."""
    if target_date is None:
        latest = max(sales["date"]) if not sales.empty else date.today()
        target_date = latest + timedelta(days=1)
    rows = []
    for item in menu_items:
        item_sales = sales[sales["item"] == item].copy()
        if item_sales.empty:
            rows.append({"item": item, "predicted_quantity": 0.0, "method": "Insufficient data — no sales history"})
            continue
        daily = item_sales.groupby("date")["quantity"].sum()
        start = min(daily.index)
        end = max(daily.index)
        idx = pd.date_range(start=start, end=end, freq="D")
        series = daily.copy()
        series.index = pd.to_datetime(series.index)
        series = series.reindex(idx, fill_value=0).astype(float)
        recent_mean = float(series.tail(7).mean())
        estimate = recent_mean
        method = "Recent 7-day average (limited history)"
        # Random Forest is used only with enough daily history to learn weekday patterns.
        if RandomForestRegressor is not None and len(series) >= 14 and int((series > 0).sum()) >= 7:
            features, labels = [], []
            vals = series.tolist()
            for i in range(7, len(vals)):
                day = series.index[i]
                features.append([day.dayofweek, day.dayofyear, vals[i-1], sum(vals[i-7:i]) / 7.0])
                labels.append(vals[i])
            if len(features) >= 7:
                model = RandomForestRegressor(n_estimators=100, random_state=42, min_samples_leaf=1)
                model.fit(features, labels)
                pred_date = pd.Timestamp(target_date)
                last_values = vals[-7:]
                x = [[pred_date.dayofweek, pred_date.dayofyear, vals[-1], sum(last_values) / 7.0]]
                estimate = float(model.predict(x)[0])
                method = "Random Forest (historical and weekday patterns)"
        rows.append({"item": item, "predicted_quantity": max(0.0, round(estimate, 2)), "method": method})
    return pd.DataFrame(rows)

def calculate_requirements(predictions, recipes):
    if recipes.empty or predictions.empty:
        return pd.DataFrame(columns=["ingredient", "predicted_requirement"])
    merged = recipes.merge(predictions[["item", "predicted_quantity"]], left_on="menu_item", right_on="item", how="left")
    merged["predicted_quantity"] = merged["predicted_quantity"].fillna(0)
    merged["requirement"] = merged["quantity_per_item"] * merged["predicted_quantity"]
    return merged.groupby("ingredient", as_index=False)["requirement"].sum().rename(columns={"requirement": "predicted_requirement"})

def purchase_table(inventory, requirements):
    result = inventory.merge(requirements, on="ingredient", how="left")
    result["predicted_requirement"] = result["predicted_requirement"].fillna(0.0)
    result["available_after_incoming"] = result["stock_quantity"] + result["incoming_stock"]
    result["recommended_order"] = (result["predicted_requirement"] + result["safety_stock"] - result["available_after_incoming"]).clip(lower=0)
    result["estimated_cost"] = result["recommended_order"] * result["purchase_cost"]
    result["low_stock"] = result["stock_quantity"] <= result["minimum_stock"]
    return result

ensure_files()
st.set_page_config(page_title="Vaishnavi Restaurant Management System", layout="wide")
st.title("Vaishnavi Restaurant Management System")
st.caption("Class 12 CBSE Artificial Intelligence Project | Code 834")
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Open section", ["Dashboard", "Sales Entry & History", "Sales Prediction", "Inventory", "Recipes", "Purchase Recommendations", "CSV Data"])
    st.divider()
    st.caption("Data is saved locally in the app's data/ folder.")

sales = read_csv(SALES_FILE, SALES_COLS)
if not sales.empty:
    try:
        sales = clean_sales(sales)
    except ValueError:
        sales = pd.DataFrame(columns=SALES_COLS)
inventory = read_csv(INVENTORY_FILE, INVENTORY_COLS)
try:
    inventory = clean_inventory(inventory)
except ValueError:
    inventory = SAMPLE_INVENTORY.copy()
recipes = read_csv(RECIPES_FILE, RECIPE_COLS)
menu = read_csv(MENU_FILE, MENU_COLS)
menu_items = menu["item"].dropna().astype(str).str.strip().loc[lambda s: s != ""].unique().tolist()

def refresh_all():
    st.rerun()

if page == "Dashboard":
    total_revenue = float(sales["revenue"].sum()) if not sales.empty else 0.0
    total_units = int(sales["quantity"].sum()) if not sales.empty else 0
    days = sales["date"].nunique() if not sales.empty else 0
    a, b, c = st.columns(3)
    a.metric("Total revenue", f"₹{total_revenue:,.2f}")
    b.metric("Items sold", f"{total_units:,}")
    c.metric("Sales days recorded", f"{days}")
    st.subheader("Sales trend")
    if sales.empty:
        st.info("No sales recorded yet. Add a sale in Sales Entry & History to start your dashboard.")
    else:
        trend = sales.groupby("date", as_index=False).agg(revenue=("revenue", "sum"), quantity=("quantity", "sum"))
        trend["date"] = pd.to_datetime(trend["date"])
        st.plotly_chart(px.line(trend, x="date", y="revenue", markers=True, title="Daily revenue (₹)"), use_container_width=True)
        left, right = st.columns(2)
        left.plotly_chart(px.bar(sales.groupby("item", as_index=False)["quantity"].sum(), x="item", y="quantity", title="Units sold by menu item"), use_container_width=True)
        right.plotly_chart(px.bar(inventory, x="ingredient", y="stock_quantity", title="Current ingredient stock"), use_container_width=True)
    st.subheader("Recent sales")
    st.dataframe(sales.sort_values("date", ascending=False).head(10), use_container_width=True, hide_index=True)

elif page == "Sales Entry & History":
    st.subheader("Add daily sales")
    with st.form("sales_form", clear_on_submit=True):
        d1, d2, d3 = st.columns(3)
        sale_date = d1.date_input("Sale date", value=date.today())
        item = d2.selectbox("Menu item", menu_items if menu_items else ["Add a menu item first"])
        quantity = d3.number_input("Quantity sold", min_value=1, step=1, value=1)
        price = st.number_input("Selling price per item (₹)", min_value=0.0, step=5.0, value=50.0)
        submitted = st.form_submit_button("Save sale", type="primary")
    if submitted:
        if not menu_items:
            st.error("Add at least one menu item in CSV Data first.")
        else:
            new = pd.DataFrame([{"date": sale_date, "item": item, "quantity": int(quantity), "price": float(price), "revenue": int(quantity) * float(price)}])
            save_frame(pd.concat([sales, new], ignore_index=True), SALES_FILE)
            st.success("Sale saved. Predictions and recommendations will use the updated data.")
            st.rerun()
    st.subheader("Sales history")
    if sales.empty:
        st.info("Sales history is empty.")
    else:
        st.dataframe(sales.sort_values("date", ascending=False), use_container_width=True, hide_index=True)
        st.download_button("Download sales CSV", sales.to_csv(index=False).encode("utf-8"), "sales.csv", "text/csv")
    st.subheader("Upload sales CSV")
    uploaded = st.file_uploader("Choose a sales CSV (date, item, quantity, price; revenue optional)", type=["csv"], key="sales_upload")
    if uploaded is not None and st.button("Validate and import sales"):
        try:
            incoming = clean_sales(pd.read_csv(uploaded))
            combined = pd.concat([sales, incoming], ignore_index=True)
            save_frame(combined, SALES_FILE)
            st.success(f"Imported {len(incoming)} sales rows.")
            st.rerun()
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            st.error(f"Could not import sales CSV: {exc}")

elif page == "Sales Prediction":
    st.subheader("Next-day sales estimates")
    if sales.empty:
        st.warning("There is no sales history yet. Record or import sales to generate data-based estimates.")
    target = (max(sales["date"]) + timedelta(days=1)) if not sales.empty else date.today() + timedelta(days=1)
    preds = next_day_predictions(sales, menu_items, target)
    st.caption(f"Target date: {target}. Predictions recalculate automatically from the saved sales file.")
    if not preds.empty:
        st.dataframe(preds.rename(columns={"item":"Menu item", "predicted_quantity":"Estimated units", "method":"Method / data sufficiency"}), use_container_width=True, hide_index=True)
        st.plotly_chart(px.bar(preds, x="item", y="predicted_quantity", color="method", title="Predicted units by menu item"), use_container_width=True)
    st.subheader("Predicted versus actual")
    if not sales.empty:
        actual = sales.groupby("item", as_index=False)["quantity"].sum().rename(columns={"quantity":"Actual units (all recorded days)"})
        compare = preds.merge(actual, on="item", how="left").fillna({"Actual units (all recorded days)":0})
        st.plotly_chart(px.bar(compare, x="item", y=["predicted_quantity", "Actual units (all recorded days)"], barmode="group", title="Next-day estimates vs total recorded units"), use_container_width=True)
        st.caption("The comparison is illustrative: predicted units are for one day, while actual units cover all recorded days.")

elif page == "Inventory":
    st.subheader("Ingredient inventory")
    st.caption("Use a consistent unit for each ingredient in both stock and its recipe quantities.")
    with st.form("inventory_add", clear_on_submit=True):
        st.markdown("**Add ingredient**")
        c1, c2, c3, c4 = st.columns(4)
        ing_name = c1.text_input("Ingredient name")
        ing_unit = c2.text_input("Unit (g, ml, pieces)")
        ing_stock = c3.number_input("Current stock", min_value=0.0, value=0.0)
        ing_min = c4.number_input("Minimum stock", min_value=0.0, value=0.0)
        c5, c6, c7 = st.columns(3)
        ing_cost = c5.number_input("Purchase cost per unit (₹)", min_value=0.0, value=0.0)
        ing_incoming = c6.number_input("Incoming stock", min_value=0.0, value=0.0)
        ing_safety = c7.number_input("Safety stock", min_value=0.0, value=0.0)
        add_ing = st.form_submit_button("Add ingredient")
    if add_ing:
        if not ing_name.strip() or not ing_unit.strip():
            st.error("Enter an ingredient name and unit.")
        elif ing_name.strip().casefold() in inventory["ingredient"].str.casefold().tolist():
            st.error("That ingredient already exists. Edit it in the table instead.")
        else:
            row = pd.DataFrame([{"ingredient":ing_name.strip(), "stock_quantity":ing_stock, "unit":ing_unit.strip(), "minimum_stock":ing_min, "purchase_cost":ing_cost, "incoming_stock":ing_incoming, "safety_stock":ing_safety}])
            save_frame(pd.concat([inventory, row], ignore_index=True), INVENTORY_FILE)
            st.success("Ingredient added.")
            st.rerun()
    if not inventory.empty:
        st.markdown("**Edit stock values below, then save.**")
        edited = st.data_editor(inventory, use_container_width=True, hide_index=True, num_rows="fixed", key="inventory_editor",
            column_config={"ingredient": st.column_config.TextColumn(disabled=True), "unit": st.column_config.TextColumn(), "stock_quantity": st.column_config.NumberColumn(min_value=0), "minimum_stock": st.column_config.NumberColumn(min_value=0), "purchase_cost": st.column_config.NumberColumn(min_value=0), "incoming_stock": st.column_config.NumberColumn(min_value=0), "safety_stock": st.column_config.NumberColumn(min_value=0)})
        col_a, col_b = st.columns(2)
        if col_a.button("Save inventory edits", type="primary"):
            try:
                cleaned = clean_inventory(edited)
                save_frame(cleaned, INVENTORY_FILE)
                st.success("Inventory changes saved.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
        delete_name = col_b.selectbox("Ingredient to delete", ["— Select —"] + inventory["ingredient"].tolist())
        if col_b.button("Delete selected ingredient"):
            if delete_name == "— Select —":
                st.error("Select an ingredient first.")
            else:
                save_frame(inventory[inventory["ingredient"] != delete_name], INVENTORY_FILE)
                st.success(f"Deleted {delete_name}. Existing recipe rows for it are retained so you can reconnect them later.")
                st.rerun()
        st.download_button("Download inventory CSV", inventory.to_csv(index=False).encode("utf-8"), "inventory.csv", "text/csv")
    st.subheader("Import inventory CSV")
    inv_upload = st.file_uploader("Choose inventory CSV with all required columns", type=["csv"], key="inventory_upload")
    if inv_upload is not None and st.button("Validate and import inventory"):
        try:
            inv_in = clean_inventory(pd.read_csv(inv_upload))
            save_frame(inv_in, INVENTORY_FILE)
            st.success(f"Imported {len(inv_in)} ingredients.")
            st.rerun()
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            st.error(f"Could not import inventory CSV: {exc}")

elif page == "Recipes":
    st.subheader("Recipe builder")
    st.write("Define how much of each ingredient one menu item uses. Recipe amounts and inventory must use the same units.")
    with st.form("recipe_form", clear_on_submit=True):
        r1, r2, r3 = st.columns(3)
        recipe_item = r1.selectbox("Menu item", menu_items if menu_items else ["No menu items"])
        recipe_ing = r2.selectbox("Ingredient", inventory["ingredient"].tolist() if not inventory.empty else ["No ingredients"])
        recipe_qty = r3.number_input("Ingredient quantity per menu item", min_value=0.001, value=1.0, step=1.0)
        recipe_submit = st.form_submit_button("Add recipe line")
    if recipe_submit:
        if recipe_item == "No menu items" or recipe_ing == "No ingredients":
            st.error("Add a menu item and ingredient first.")
        else:
            newrow = pd.DataFrame([{"menu_item":recipe_item, "ingredient":recipe_ing, "quantity_per_item":recipe_qty}])
            save_frame(pd.concat([recipes, newrow], ignore_index=True), RECIPES_FILE)
            st.success("Recipe line saved.")
            st.rerun()
    if not recipes.empty:
        st.markdown("**Edit or remove recipe lines**")
        edited_recipes = st.data_editor(recipes, use_container_width=True, hide_index=True, num_rows="dynamic", key="recipe_editor",
            column_config={"menu_item": st.column_config.SelectboxColumn(options=menu_items), "ingredient": st.column_config.SelectboxColumn(options=inventory["ingredient"].tolist()), "quantity_per_item": st.column_config.NumberColumn(min_value=0.001)})
        if st.button("Save recipe changes", type="primary"):
            try:
                er = edited_recipes.copy()
                er["quantity_per_item"] = pd.to_numeric(er["quantity_per_item"], errors="raise")
                if (er["quantity_per_item"] <= 0).any() or er[["menu_item","ingredient"]].isna().any().any():
                    raise ValueError("Choose a menu item and ingredient and use a positive quantity.")
                save_frame(er[RECIPE_COLS], RECIPES_FILE)
                st.success("Recipes saved.")
                st.rerun()
            except (ValueError, KeyError) as exc:
                st.error(f"Could not save recipes: {exc}")
        st.download_button("Download recipes CSV", recipes.to_csv(index=False).encode("utf-8"), "recipes.csv", "text/csv")
    else:
        st.info("No recipes yet. Add recipe lines above.")

elif page == "Purchase Recommendations":
    st.subheader("Ingredient purchase recommendations")
    target = (max(sales["date"]) + timedelta(days=1)) if not sales.empty else date.today() + timedelta(days=1)
    preds = next_day_predictions(sales, menu_items, target)
    requirements = calculate_requirements(preds, recipes)
    purchases = purchase_table(inventory, requirements)
    st.caption(f"Based on estimated sales for {target}. Update sales, recipes, or stock to refresh this calculation.")
    if sales.empty:
        st.warning("No sales history: sales estimates are zero until you enter or import sales. Recommendations still include configured safety stock.")
    if recipes.empty:
        st.warning("No recipe lines found. Add recipes to calculate ingredient requirements.")
    if purchases.empty:
        st.info("Add ingredients to inventory to see purchase recommendations.")
    else:
        st.metric("Estimated purchase cost", f"₹{purchases['estimated_cost'].sum():,.2f}")
        view = purchases[["ingredient","unit","stock_quantity","incoming_stock","predicted_requirement","safety_stock","recommended_order","purchase_cost","estimated_cost","low_stock"]].rename(columns={
            "ingredient":"Ingredient","unit":"Unit","stock_quantity":"Current stock","incoming_stock":"Incoming stock","predicted_requirement":"Predicted requirement","safety_stock":"Safety stock","recommended_order":"Recommended order","purchase_cost":"Cost per unit (₹)","estimated_cost":"Estimated cost (₹)","low_stock":"Low-stock warning"})
        st.dataframe(view, use_container_width=True, hide_index=True)
        lows = purchases[purchases["low_stock"]]
        if not lows.empty:
            st.warning("Low stock: " + ", ".join(lows["ingredient"].tolist()))
        st.download_button("Download purchase recommendations", view.to_csv(index=False).encode("utf-8"), "purchase_recommendations.csv", "text/csv")
        st.plotly_chart(px.bar(purchases, x="ingredient", y="recommended_order", title="Recommended order quantity by ingredient"), use_container_width=True)

elif page == "CSV Data":
    st.subheader("Menu items")
    st.write("Add menu items here. The menu is saved in menu.csv and used by sales and recipe forms.")
    with st.form("menu_form", clear_on_submit=True):
        menu_name = st.text_input("New menu item name")
        menu_submit = st.form_submit_button("Add menu item")
    if menu_submit:
        if not menu_name.strip():
            st.error("Menu item name cannot be blank.")
        elif menu_name.strip().casefold() in [x.casefold() for x in menu_items]:
            st.error("That menu item already exists.")
        else:
            save_frame(pd.concat([menu, pd.DataFrame([{"item":menu_name.strip()}])], ignore_index=True).drop_duplicates("item"), MENU_FILE)
            st.success("Menu item added.")
            st.rerun()
    st.dataframe(menu, use_container_width=True, hide_index=True)
    st.download_button("Download menu CSV", menu.to_csv(index=False).encode("utf-8"), "menu.csv", "text/csv")
    st.divider()
    st.subheader("Data files")
    st.write("The app creates and updates these files in the local `data/` folder:")
    st.code("data/sales.csv\ndata/inventory.csv\ndata/recipes.csv\ndata/menu.csv")
    st.info("Back up this folder regularly. CSV storage is simple and suitable for a school project, but it is not a multi-user database.")
