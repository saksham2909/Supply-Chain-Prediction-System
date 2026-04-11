============================================================
   SUPPLY CHAIN PROJECT - KAISE CHALAYEIN
============================================================

STEP 1: VS Code kholo
   File -> Open Folder -> "Supply Chain Delay Prediction System"

STEP 2: Terminal kholo
   Ctrl + ~ dabaao

STEP 3: Virtual Env Activate karo
   E:\Data Analyst Projects\venv\Scripts\activate
   (venv) dikhne lage toh activate ho gaya

STEP 4: Main Analysis run karo
   python supply_chain_analysis.py

STEP 5: Flask Web App run karo (optional)
   python flask_app.py
   Browser mein ja: http://localhost:5000

============================================================
   FILES KA KAM KYA HAI
============================================================

supply_chain_analysis.py  -> Main project (EDA + ML)
flask_app.py              -> Web app (prediction)
supply_chain_data.csv     -> Raw dataset
supply_chain_powerbi.csv  -> Power BI ke liye data
eda_dashboard.png         -> EDA charts (run hone ke baad banta hai)
ml_evaluation.png         -> ML charts (run hone ke baad banta hai)

============================================================
   ERRORS AUR FIX
============================================================

ModuleNotFoundError  -> pip install pandas numpy scikit-learn matplotlib seaborn flask
venv activate nahi   -> Step 3 wali command dobara chalao
Port already in use  -> flask_app.py mein port=5001 kar do

============================================================
