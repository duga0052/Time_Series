

Time Series
│
├── data/
│   ├── __init__.py
│   ├── AAPL.csv
│   └── data_loader.py
│
├── preprocessing/
│   ├── __init__.py
│   └── data_preprocessing.py
│
├── feature/
│   ├── __init__.py
    ├──train_test_split.py
│   └── feature_selection.py
│
├── models/
│   ├── __init__.py
│   ├── arima_model.py
│   ├── arimax_model.py
│   ├── sarima_model.py
│   ├── sarimax_model.py
│   └── xgboost_model.py
│
├── prediction/
│   ├── __init__.py
│   └── backtest.py
    └──prediction.py
│
├── visualization/
│   ├── __init__.py
│   └── plots.py
├── main.py
├── requirements.txt
└── README.md

Steps to Push code from VS code to Github.
First authenticate your githib account and integrate with VS code. Click on the source control icon and complete the setup.
1. Click terminal and open new terminal
2. git config --global user.name "Swapnilin"
3. git config --global user.email swapnilforcat@gmail.com
4. git init
5. git add .
6. git commit -m "Your commit message"