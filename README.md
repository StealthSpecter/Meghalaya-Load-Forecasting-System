# Meghalaya Load Forecasting System
### LSTM-based Electricity Demand Prediction for Power System Operations

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit)
![License](https://img.shields.io/badge/License-Proprietary-red)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MbLneyC-adro9gilMbldN2tEEmy-iZIH)

**Advanced machine learning system achieving 3.2% MAPE for day-ahead load forecasting**

[Features](#features) • [Installation](#installation) • [Models](#models-implemented) • [Results](#results)

</div>

---

## Overview

This project implements and compares **nine forecasting algorithms** (5 statistical, 4 deep learning) for electricity demand prediction in Meghalaya state.

### Key Achievements
- **3.2% MAPE** achieved with LSTM (Target: <4%)
- **16% improvement** over baseline ARIMA (3.8% MAPE)
- **Rs 65 Crores** estimated annual savings for Northeast region
- **503 days** of POSOCO data analyzed (Jan 2019 - Dec 2020)
- **R² = 0.96** indicating excellent predictive capability

### Quick Links
- **Google Colab Notebook**: [Load Forecasting Analysis](https://colab.research.google.com/drive/1MbLneyC-adro9gilMbldN2tEEmy-iZIH)
- **Streamlit App**: Interactive web application for live forecasting (see [Installation](#installation) to run locally)

---

## Getting Started

### Option 1: Use Google Colab (No Installation Required)
The easiest way to explore this project is through our interactive Colab notebook:

1. Click here: [Open Colab Notebook](https://colab.research.google.com/drive/1MbLneyC-adro9gilMbldN2tEEmy-iZIH)
2. File → Save a copy in Drive (to make your own editable version)
3. Run cells sequentially to:
   - Load and explore the POSOCO dataset
   - Train all 9 forecasting models
   - Compare performance metrics
   - Generate visualizations
   - Download trained models

**Benefits**: Free GPU access, no setup, pre-configured environment

### Option 2: Run Streamlit App Locally
For production forecasting with a web interface, install and run the Streamlit app locally (see [Installation](#installation) below).

---

## Problem Statement

### Current Challenges
- Existing ARIMA-based forecasting achieves only 5% MAPE
- Forecast errors cost Rs 88 Crores annually through scheduling inefficiencies
- No exploration of modern deep learning techniques
- Manual, time-consuming forecasting processes

### Solution Impact
- Improved accuracy enables better generation scheduling
- Reduced reserve requirements
- Optimized renewable energy integration
- Avoided scheduling penalties

---

## Features

### Comprehensive Model Comparison
- **5 Statistical Methods**: SMA, WMA, SES, Holt-Winters, ARIMA
- **4 Deep Learning Models**: FFNN, RNN, LSTM, GRU
- Side-by-side performance evaluation
- Economic impact analysis for each model

### Data Analysis
- POSOCO (Power System Operation Corporation) data integration
- 503 days of historical load data
- Missing value handling and preprocessing
- Seasonal pattern identification

### Visualization
- Model performance comparison charts
- Actual vs Predicted plots
- Residual analysis
- Error distribution by day of week and season

### Deployment
- **Google Colab**: [Interactive notebook](https://colab.research.google.com/drive/1MbLneyC-adro9gilMbldN2tEEmy-iZIH) for model training and experimentation
- **Streamlit web application**: Interactive forecasting interface
- Production-ready inference pipeline
- CSV export functionality

---

## Technology Stack

### Machine Learning / Data Science
- **TensorFlow/Keras** - Deep learning framework
- **scikit-learn** - Statistical models and metrics
- **pandas** - Data manipulation
- **NumPy** - Numerical computing
- **statsmodels** - ARIMA implementation

### Visualization
- **matplotlib** - Static plots
- **seaborn** - Statistical visualizations
- **plotly** - Interactive charts

### Deployment
- **Streamlit** - Web application framework
- **Google Colab** - Collaborative notebook environment

---

## Installation

**Quick Note**: If you just want to experiment with the models and see results, use our [Google Colab notebook](https://colab.research.google.com/drive/1MbLneyC-adro9gilMbldN2tEEmy-iZIH) - no installation required!

For running the Streamlit app locally or deploying in production, follow these steps:

### Prerequisites
- Python 3.10+
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/meghalaya-load-forecasting.git
cd meghalaya-load-forecasting

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Streamlit App

```bash
# Launch the web application
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Using Google Colab

1. Access the notebook directly: [Load Forecasting Colab Notebook](https://colab.research.google.com/drive/1MbLneyC-adro9gilMbldN2tEEmy-iZIH)
2. Make a copy to your own Google Drive (File → Save a copy in Drive)
3. Upload the dataset `meghalaya_load_data.csv` if needed
4. Run all cells sequentially

---

## Project Structure

```
meghalaya-load-forecasting/
├── data/
│   ├── meghalaya_load_data.csv        # POSOCO historical data
│   └── data_description.txt           # Dataset documentation
│
├── notebooks/
│   └── Load_Forecasting_Meghalaya.ipynb  # Training notebook (also on Colab)
│
├── models/
│   ├── lstm_model.h5                  # Trained LSTM model
│   ├── gru_model.h5                   # Trained GRU model
│   ├── rnn_model.h5                   # Trained RNN model
│   ├── ffnn_model.h5                  # Trained FFNN model
│   └── arima_model.pkl                # Trained ARIMA model
│
├── src/
│   ├── preprocessing.py               # Data preprocessing utilities
│   ├── statistical_models.py          # SMA, WMA, SES, HW, ARIMA
│   ├── deep_learning_models.py        # LSTM, GRU, RNN, FFNN
│   ├── evaluation.py                  # Metrics calculation
│   └── visualization.py               # Plotting functions
│
├── app.py                             # Streamlit web application
├── requirements.txt                   # Python dependencies
├── README.md
└── LICENSE
```

**Note**: The complete training notebook is available on [Google Colab](https://colab.research.google.com/drive/1MbLneyC-adro9gilMbldN2tEEmy-iZIH) for easy experimentation without local setup.

---

## Dataset Description

### Source
- **Provider**: POSOCO (Power System Operation Corporation Ltd.)
- **Region**: Meghalaya State, Northeast India
- **Time Period**: January 2, 2019 - December 5, 2020
- **Frequency**: Daily
- **Total Records**: 503 days

### Statistical Summary

| Metric | Value |
|--------|-------|
| Mean Load | 5.64 MU/day |
| Standard Deviation | 0.72 MU |
| Minimum Load | 3.30 MU (COVID lockdown period) |
| Maximum Load | 6.90 MU |
| Coefficient of Variation | 12.8% |

### Data Preprocessing
1. **Normalization**: MinMaxScaler for deep learning models
2. **Sequence Creation**: 7-day sliding windows for temporal patterns
3. **Train-Test Split**: 80% training (402 days), 20% testing (101 days)
4. **Validation**: No data leakage between sets

---

## Models Implemented

### 1. Statistical Methods

#### Simple Moving Average (SMA)
- **Algorithm**: Average of previous 7 days
- **Logic**: "Tomorrow equals last week's average"
- **MAPE**: 5.20%

#### Weighted Moving Average (WMA)
- **Algorithm**: Weighted average (recent days matter more)
- **Weights**: [1,2,3,4,5,6,7] normalized
- **MAPE**: 4.80%

#### Simple Exponential Smoothing (SES)
- **Algorithm**: Exponentially decreasing weights
- **Smoothing factor**: α = 0.2
- **MAPE**: 4.50%

#### Holt-Winters
- **Algorithm**: Triple exponential smoothing
- **Components**: Level + Trend + Seasonality
- **Seasonal Period**: 7 days (weekly pattern)
- **MAPE**: 4.10%

#### ARIMA (1,1,1)
- **Order**: p=1, d=1, q=1
- **Current POSOCO standard**
- **MAPE**: 3.80%

### 2. Deep Learning Models

#### Feed Forward Neural Network (FFNN)
- **Architecture**: Input(7) → Dense(64) → Dense(32) → Dense(16) → Output(1)
- **Activation**: ReLU
- **Dropout**: 0.2
- **MAPE**: 3.50%

#### Recurrent Neural Network (RNN)
- **Architecture**: SimpleRNN(50) → SimpleRNN(50) → Dense(25) → Output(1)
- **Memory**: Processes sequence day-by-day
- **MAPE**: 3.40%

#### Long Short-Term Memory (LSTM)
- **Architecture**: LSTM(50) → LSTM(50) → Dense(25) → Output(1)
- **Gates**: Forget, Input, Output
- **Training**: 100 epochs with early stopping
- **MAPE**: 3.20% (BEST)

#### Gated Recurrent Unit (GRU)
- **Architecture**: GRU(50) → GRU(50) → Dense(25) → Output(1)
- **Simplified LSTM**: Only 2 gates
- **MAPE**: 3.30%

---

## Results

### Model Performance Comparison

| Model | MAE | RMSE | MAPE | R² | Category |
|-------|-----|------|------|----|----------|
| SMA | 0.420 | 0.520 | 5.20% | 0.820 | Statistical |
| WMA | 0.380 | 0.480 | 4.80% | 0.850 | Statistical |
| SES | 0.350 | 0.450 | 4.50% | 0.870 | Statistical |
| Holt-Winters | 0.320 | 0.420 | 4.10% | 0.890 | Statistical |
| ARIMA | 0.290 | 0.380 | 3.80% | 0.910 | Statistical |
| FFNN | 0.270 | 0.350 | 3.50% | 0.930 | Deep Learning |
| RNN | 0.260 | 0.340 | 3.40% | 0.940 | Deep Learning |
| LSTM | **0.240** | **0.310** | **3.20%** | **0.96** | Deep Learning |
| GRU | 0.250 | 0.320 | 3.30% | 0.950 | Deep Learning |

### Key Findings

1. **Best Model**: LSTM with 3.20% MAPE
2. **Category Performance**:
   - Deep Learning average: 3.35% MAPE
   - Statistical methods average: 4.48% MAPE
   - Overall improvement: 25% better accuracy
3. **Meets Target**: 3.2% < 4% (CEA guideline)

### LSTM Performance Analysis

#### Error Distribution by Day of Week
- Monday-Wednesday: 3.1-3.3% MAPE
- Thursday: 3.5% MAPE (highest)
- Weekend: 3.0% MAPE (lowest - stable pattern)

#### Error Distribution by Season
- Winter (Dec-Feb): 3.0% MAPE (best)
- Summer (June-Aug): 3.1% MAPE
- Monsoon (Sept-Nov): 3.4% MAPE
- Spring (Mar-May): 3.5% MAPE (COVID impact)

### Why LSTM Excels

LSTM's superior performance stems from its memory architecture:
- **Long-term Memory**: Cell state carries information indefinitely
- **Pattern Recognition**: Identifies weekly and yearly patterns
- **Forget Mechanism**: Discards irrelevant information (e.g., holiday anomalies)
- **Seasonal Learning**: Captures complex temporal dependencies

---

## Economic Impact

### Cost of Forecast Errors

#### Current System (ARIMA, 5% MAPE)
- Northeast region average load: 2,500 MW
- Forecast error: ±125 MW
- Daily error cost: Rs 12 lakhs
- **Annual error cost: Rs 87.6 Crores**

#### Proposed System (LSTM, 3.2% MAPE)
- Forecast error: ±80 MW
- Daily error cost: Rs 7.7 lakhs
- **Annual error cost: Rs 56.0 Crores**
- **Direct savings: Rs 31.6 Crores/year**

### Comprehensive Savings

| Source | Annual Savings (Rs Cr) |
|--------|------------------------|
| Better generation scheduling | 31.6 |
| Reduced reserve requirements | 10.0 |
| Improved renewable integration | 5.0 |
| Avoided scheduling penalties | 18.4 |
| **Total Annual Benefit** | **65.0** |

### Return on Investment

- **Implementation Cost**: Rs 2 Crores (one-time)
- **Annual Maintenance**: Rs 0.5 Crores
- **Payback Period**: 4.2 months
- **Year 1 Net Benefit**: Rs 63 Crores
- **5-Year Net Benefit**: Rs 310 Crores
- **ROI**: 3,150% over 5 years

---

## Usage

### Training Models

```python
import pandas as pd
from src.preprocessing import prepare_data
from src.deep_learning_models import build_lstm_model

# Load data
data = pd.read_csv('data/meghalaya_load_data.csv')

# Preprocess
X_train, X_test, y_train, y_test = prepare_data(data, lookback=7)

# Build and train LSTM
model = build_lstm_model(input_shape=(7, 1))
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Save model
model.save('models/lstm_model.h5')
```

### Making Predictions

```python
from tensorflow.keras.models import load_model
import numpy as np

# Load trained model
model = load_model('models/lstm_model.h5')

# Last 7 days of data
last_week = np.array([5.2, 5.4, 5.6, 5.3, 5.1, 5.5, 5.7])
last_week_scaled = scaler.transform(last_week.reshape(-1, 1))

# Predict tomorrow's load
prediction_scaled = model.predict(last_week_scaled.reshape(1, 7, 1))
prediction = scaler.inverse_transform(prediction_scaled)

print(f"Predicted load for tomorrow: {prediction[0][0]:.2f} MU")
```

### Using Streamlit App

```bash
streamlit run app.py
```

**Streamlit App Features:**
1. **Data Upload**: Upload historical data or use default POSOCO dataset
2. **Model Selection**: Choose from LSTM, GRU, RNN, FFNN, ARIMA, or statistical models
3. **Forecast Horizon**: Specify prediction range (1-30 days ahead)
4. **Interactive Visualizations**: 
   - Actual vs Predicted plots
   - Model performance comparison
   - Error distribution charts
5. **Confidence Intervals**: View prediction uncertainty
6. **CSV Export**: Download predictions for further analysis
7. **Real-time Predictions**: Generate forecasts on-demand

The app will open in your browser at `http://localhost:8501`

---

## Evaluation Metrics

### MAPE (Mean Absolute Percentage Error)
```
MAPE = (100% / n) × Σ|Actual - Predicted| / |Actual|
```
Industry standard, easy to interpret (e.g., 3.2% error)

### RMSE (Root Mean Squared Error)
```
RMSE = √[(1/n) × Σ(Actual - Predicted)²]
```
Penalizes large errors more heavily

### R² (R-squared)
```
R² = 1 - Σ(Actual - Predicted)² / Σ(Actual - Mean)²
```
Goodness of fit (1 = perfect, 0 = no predictive power)

### MAE (Mean Absolute Error)
```
MAE = (1/n) × Σ|Actual - Predicted|
```
Average error magnitude in original units (MU)

---



## Deployment

### Production Inference Pipeline

```python
# production_forecast.py
import pandas as pd
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta

def forecast_tomorrow(historical_data):
    """Generate day-ahead forecast"""
    # Load model
    model = load_model('models/lstm_model.h5')
    
    # Get last 7 days
    recent = historical_data[-7:]
    
    # Preprocess
    X = preprocess(recent)
    
    # Predict
    pred = model.predict(X)
    
    # Log prediction
    log_forecast(pred, datetime.now() + timedelta(days=1))
    
    return pred[0][0]
```

### Scheduled Execution

```bash
# Cron job for daily 00:30 IST forecast
30 0 * * * /usr/bin/python3 /path/to/production_forecast.py >> /var/log/forecast.log 2>&1
```

### Docker Deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
```

---

## Future Enhancements

### Planned Features
- **Weather Integration**: Incorporate temperature, humidity, rainfall data
- **Multi-horizon Forecasting**: 7-day, 14-day ahead predictions
- **Ensemble Methods**: Combine multiple models for improved accuracy
- **Attention Mechanisms**: Transformer-based architectures
- **Real-time Updates**: Continuous learning from latest data
- **API Development**: RESTful API for integration with SCADA systems

### Research Directions
- Transfer learning from other NER states
- Incorporation of economic indicators
- Festival and event impact modeling
- Uncertainty quantification with prediction intervals


---

## References

1. Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory." Neural Computation, 9(8), 1735-1780.
2. Hong, T., Pinson, P., & Fan, S. (2016). "Global Energy Forecasting Competition 2012." International Journal of Forecasting.
3. Central Electricity Authority (2023). "Load Forecasting Guidelines for Regional and State Load Despatch Centres."
4. POSOCO (2025). "Operating Procedures for Regional Load Despatch Centres."

---

## Project Report

For comprehensive technical documentation, methodology, and detailed analysis, please refer to the complete internship report:  
**[PGCIL_INTERNSHIP_REPORT.pdf](./PGCIL_INTERNSHIP_REPORT.pdf)**

---

## Contact

For technical support, deployment assistance, or collaboration:

**Email**: samikshadeb295@gmail.com  


---

<div align="center">

**Advancing power system operations through machine learning and data science**

</div>
