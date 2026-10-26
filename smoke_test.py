import sys, types, importlib.util, tempfile
from pathlib import Path
import pandas as pd
# Streamlit stub lets us import pure functions without launching a browser.
st = types.ModuleType("streamlit")
st.set_page_config=lambda **kw: None
st.title=lambda *a,**kw: None
st.caption=lambda *a,**kw: None
class DummyContext:
 def __enter__(self): return self
 def __exit__(self,*a): return False
st.sidebar=DummyContext()
st.header=lambda *a,**kw:None
st.radio=lambda *a,**kw:"Dashboard"
st.divider=lambda *a,**kw:None
st.write=lambda *a,**kw:None
st.info=lambda *a,**kw:None
st.subheader=lambda *a,**kw:None
st.columns=lambda n:[types.SimpleNamespace(metric=lambda *a,**kw:None) for _ in range(n)]
st.metric=lambda *a,**kw:None
st.dataframe=lambda *a,**kw:None
st.plotly_chart=lambda *a,**kw:None
st.warning=lambda *a,**kw:None
st.download_button=lambda *a,**kw:None
st.code=lambda *a,**kw:None
st.markdown=lambda *a,**kw:None
st.form=lambda *a,**kw:types.SimpleNamespace(__enter__=lambda s:s,__exit__=lambda *a:False)
st.text_input=lambda *a,**kw:""
st.number_input=lambda *a,**kw:0.0
st.date_input=lambda *a,**kw:__import__('datetime').date.today()
st.selectbox=lambda *a,**kw:(a[1][0] if len(a)>1 and isinstance(a[1],list) else "")
st.form_submit_button=lambda *a,**kw:False
st.button=lambda *a,**kw:False
st.file_uploader=lambda *a,**kw:None
st.data_editor=lambda x,*a,**kw:x
st.rerun=lambda:None
st.column_config=types.SimpleNamespace(TextColumn=lambda *a,**kw:None,NumberColumn=lambda *a,**kw:None,SelectboxColumn=lambda *a,**kw:None)
st.set_page_config=lambda **kw:None
sys.modules["streamlit"]=st
# plotly.express import is available in runtime; otherwise skip import-based test.
spec=importlib.util.spec_from_file_location("app", sys.argv[1])
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
from datetime import date, timedelta
sales=pd.DataFrame([
 {"date":date(2026,10,1)+timedelta(days=i),"item":"Veg Pizza","quantity":i%4+1,"price":100.0,"revenue":(i%4+1)*100.0}
 for i in range(20)
])
pred=mod.next_day_predictions(sales,["Veg Pizza","No History"],date(2026,10,21))
assert list(pred["item"])==["Veg Pizza","No History"]
assert pred.loc[pred.item=="Veg Pizza","predicted_quantity"].iloc[0]>=0
assert "no sales history" in pred.loc[pred.item=="No History","method"].iloc[0].lower()
recipes=pd.DataFrame([{"menu_item":"Veg Pizza","ingredient":"Flour","quantity_per_item":200.0},{"menu_item":"Veg Pizza","ingredient":"Cheese","quantity_per_item":100.0}])
req=mod.calculate_requirements(pred,recipes)
assert req.loc[req.ingredient=="Flour","predicted_requirement"].iloc[0] == 200*pred.loc[pred.item=="Veg Pizza","predicted_quantity"].iloc[0]
inv=pd.DataFrame([{"ingredient":"Flour","stock_quantity":100.0,"unit":"g","minimum_stock":10.0,"purchase_cost":0.04,"incoming_stock":20.0,"safety_stock":30.0}])
p=mod.purchase_table(inv,req)
assert p.recommended_order.iloc[0] == max(0, p.predicted_requirement.iloc[0]+30-100-20)
assert (p.recommended_order>=0).all()
clean=mod.clean_sales(pd.DataFrame([{"date":"2026-10-20","item":"Tea","quantity":2,"price":15}]))
assert clean.revenue.iloc[0] == 30
try:
 mod.clean_sales(pd.DataFrame([{"date":"bad","item":"Tea","quantity":-1,"price":15}]))
 raise AssertionError("invalid sale should fail")
except ValueError: pass
print("PASS: prediction, insufficient-data labeling, recipe requirements, non-negative purchase formula, revenue calculation, and invalid-sales validation")
