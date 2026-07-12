# Serenair-Flight-Trajectory-Detection

**Real-time Flight Trajectory Prediction with Machine Learning Models**

Serenair is a full-stack machine learning application that harnesses the power of LSTM neural networks to predict aircraft position at next-instance and 10~ minutes ahead in real time using live ADS-B and wind data.
Furthermore, it makes use of DBSCAN clustering algorithm on these predictions to identify aircraft density/traffic clusters 10~ minutes ahead in time.

**www.serenair.live**

<img width="1763" height="2069" alt="Screenshot_2-7-2026_125954_serenair live" src="https://github.com/user-attachments/assets/bb3e0a6f-7935-4e1c-8890-020b21febc65" />


---

## Features:
- **Next Instance Prediction**: Predict aircraft position for next instance (~10 secs).
- **Extended Horizon Prediction**: Predict full ~10 minute trajectory paths for aircraft.
- **Traffic Density/Convergence Forecasting**: Identify high-density airspace regions ~10 minutes in advance.
- **Real-time Dashboard**: Get real time insights about live aircrafts.
- **Multi-Continent Support**: Coverage for all continents.
- **Weather Integration**: Integration with wind data for better and reliable forecasting.
- **Interactive Maps**: Bokeh-powered visualisation of flight trajectories and prediction maps.

---
<div align="center">

**Crowd Density Prediction**

*Identify high-density airspace regions and flight convergence zones ~10 minutes in advance using DBSCAN clustering algorithm on trajectory prediction*

<img width="1000" height="530" alt="Crowd Density Prediction Map" 
     src="https://github.com/user-attachments/assets/e0191a1a-0a6c-4e20-9370-8bc45e69b9de" />
</div>

---

## Architecture and Tech Stack:
### Data Pipeline
```
  OpenSky Network API (live ADS-B data)
                 ↓
    Data cleaning and processing
                 ↓
    Open-Meteo Weather Integration
                 ↓
        Feature Engineering
                 ↓
        PostgreSQL database
                 ↓
  Predictions (with Redis caching)
```

### Tech Stack
```
-Backend: Flask, Pytorch, Pandas, Numpy, Scikit-learn

-Frontend: Bokeh, Plotly, D3.js

-Database and Caching: PostgreSQL, Redis

-Infrastructure: Docker, Nginx, MLflow

-API: OpenSky, Open-Meteo
```
---

## Model Architectures and Hyperparameters:
## LSTM (Next Instance Prediction)
Predicts aircraft position at t+1 (~10 seconds ahead)
 
```
Model: LSTMModelV1
Input: 10-step window of 21 features
Output: 2 values (Δlat, Δlon)
 
Hyperparameters:
- Batch Size: 128
- Epochs: 100
- Hidden Size: 64
- Num Layers: 2
- Learning Rate: 0.001
- Optimizer: Adam
- Loss Function: Huber Loss (δ=1.0)
- Weight Decay: 1e-05
```
 
## Seq2Seq LSTM (10-Minute Prediction)
Predicts full 10-minute trajectory (t+6 to t+60, one prediction per minute)
 
```
Model: Seq2SeqLSTMV1
Encoder: LSTM processing 10-step history
Decoder: LSTM generating 10 future steps
 
Hyperparameters:
- Batch Size: 128
- Epochs: 100
- Hidden Size: 64
- Num Layers: 2
- Learning Rate: 0.001
- Optimizer: Adam
- Loss Function: Huber Loss (δ=1.0)
- Weight Decay: 1e-05
```


---

## Performance and Results:
## Evaluation Metric
- **Haversine Distance**: Measures real-world prediction error in meters
- Single time step = 10 seconds
- t+1 = immediate next observation (~10 seconds)
- t+60 = 10 minutes ahead
## Baseline Comparisons
 
| Baseline | Horizon | Mean Error |
|----------|---------|-----------|
| Linear Baseline (Constant Velocity)| t+1 | 1,543.60 m |
| Naive Repeat Baseline (Constant Velocity and heading) | t+60 | 2,196.40 m |
 
## Model Performance
 
| Model | Horizon | Mean Error | Improvement |
|-------|---------|-----------|-------------|
| LSTM | t+1 | 322.0 | 79.2% vs baseline |
| Seq2Seq LSTM | t+60 | 763.60 | 65.3% vs baseline |
 
## Weather Integration Impact
Weather integration (wind at 250 hPa + temperature at 2m) improved prediction accuracy by **20-27%** compared to ADS-B-only models.


---

## Working:

## 1. Data Collection & Integration

The system collects global ADS-B flight data in real-time via the OpenSky Network API. Data collection is triggered automatically when users visit the application, initiating background workers that continuously poll the API at 10-second intervals.

**Data Pipeline:**
- Flight data is ingested into PostgreSQL for persistent storage
- Old records (>15 minutes) are automatically pruned to maintain fresh datasets
- Meteorological data (wind, temperature) from Open-Meteo is fetched and temporally aligned with flight observations
- All data undergoes automated quality checks to remove invalid or corrupted records

## 2. Real-Time Prediction Pipeline

When users navigate to prediction pages, the `/api/wakeup` endpoint triggers the background preprocessing and inference workflow:

**Preprocessing Steps:**
1. Data cleaning: Removes stale records, handles missing values, filters anomalies
2. Feature engineering: Derives temporal (hour encoding), kinematic (acceleration, turn rate), and contextual features from raw ADS-B data
3. Sequence creation: Constructs 10-timestep sliding windows (100-second history) for each aircraft trajectory
4. Feature scaling: Applies pre-trained StandardScaler to input features and RobustScaler to target variables (delta coordinates)

**Inference & Caching:**
- Models run inference on preprocessed sequences per continent
- Predictions are cached in Redis (~15-second TTL) to minimize latency and redundant computation
- Results include next-instance positions (t+1) and 10-minute trajectory forecasts (t+6 to t+60)

**Visualization:**
- Bokeh-powered interactive maps render predicted trajectories on geographic base layers
- DBSCAN clustering identifies traffic convergence zones from multi-step predictions
- Maps are saved as standalone HTML files for download and offline use

---

# Dataset & Features
 
## Data Source
 
| Property | Details |
|----------|---------|
| **Source** | OpenSky Network API (ADS-B broadcast data) |
| **Collection Date** | April 4, 2026 |
| **Dataset Size** | ~1 million observations |
| **Polling Frequency** | 10-second intervals |
| **Geographic Coverage** | Global (all regions) |
| **Weather Data** | Open-Meteo meteorological API |
 
## Feature Set (21 Total)
 
### Raw ADS-B Features (8)
- **Position:** Latitude, longitude, barometric altitude
- **Kinematics:** Ground speed (velocity), true track (heading), vertical rate
- **Metadata:** Timestamp, aircraft mass category
### Engineered Features (6)
Derived from raw observations to capture temporal and dynamic patterns:
- **Temporal Encoding:** Δt (time since last observation), hour_sin, hour_cos (cyclic hour encoding)
- **Kinematic Derivatives:** Acceleration (dv/dt), turn rate (dtrack/dt), climb phase classification
### Meteorological Features (3)
Critical for capturing atmospheric influences on flight dynamics:
- **Wind Speed at 250 hPa:** Jet stream-level wind velocity (affects cruise-phase ground speed)
- **Wind Direction at 250 hPa:** Directional component critical for trajectory modeling
- **Surface Temperature (2m):** Air density proxy affecting takeoff and climb performance
### Target Variables (2)
- **Δ Latitude:** Change in latitude from current to next timestep
- **Δ Longitude:** Change in longitude from current to next timestep

---
 
# Data Preprocessing & Training
 
## Preprocessing Pipeline
 
**Scaling Strategy:**
- **Features (X):** StandardScaler (μ=0, σ=1) for neural network stability
- **Targets (y):** RobustScaler (median-centered, IQR-normalized) to handle outliers in coordinate deltas
**Sequence Construction:**
- Sliding window of 10 timesteps (100 seconds of history)
- Each window creates one training sample for t+1 prediction
- Extended windows (10 steps × 10 steps) for multi-step t+60 prediction
**Train/Test Split:**
- **Strategy:** Group Shuffle Split at aircraft level (ICAO24 identifier)
- **Ratio:** 90% training, 10% validation
- **Purpose:** Prevents data leakage by keeping entire flight trajectories in single partition
**Validation Approach:**
- Time-series aware cross-validation (no future-to-past leakage)
- Stratified sampling ensures representation of flight types and regions
- Early stopping based on Haversine distance metric on validation set

---
 
# Key Findings
 
1. **Weather Integration**: Wind data at cruise altitude (250 hPa) is critical for trajectory prediction; integration improves accuracy by 20-27%
2. **Multi-Step Prediction**: Seq2Seq architecture effectively learns temporal dependencies over 10-minute horizons
3. **Real-World Applicability**: DBSCAN clustering transforms point predictions into actionable traffic density maps for air traffic control
4. **Scalability**: System processes global flights in real-time with Redis caching (~15-second refresh)

---
 
# Future Work
- Anomaly detection in trajectory patterns
- Integration with actual ATC data for validation
- Extended prediction horizon (20+ minutes)
---
 
# References
- Sutskever et al. (2014) - Seq2Seq Learning
- Hochreiter & Schmidhuber (1997) - LSTM Networks
- Ester et al. (1996) - DBSCAN Algorithm
- Schäfer et al. (2014) - OpenSky Network
- Huber (1964) - Robust Loss Functions
 
---
 
# Getting Started
 
## Prerequisites
```
Python 3.10+
PyTorch 2.0+
PostgreSQL 13+
Redis 6.0+
Docker (for containerized deployment)
```
 
## Installation
```bash
git clone https://github.com/yashdeshpandex1/Serenair-Flight-Trajectory-Prediction.git
cd Serenair
pip install -r requirements.txt
```
 
## Configuration
```bash
# Create .env file with API credentials
OPENSKY_USERNAME=your_username
OPENSKY_PASSWORD=your_password
DATABASE_URL=postgresql://user:pass@localhost/serenair
REDIS_HOST=localhost
```
 
## Run Application
```bash
# Development
flask run
 
# Production (Docker)
docker-compose up -d
```
 
Access at `http://localhost:5000`
 
---

---

## Contact me:
- **LinkedIn**: https://www.linkedin.com/in/yash-deshpandeb6
- **Email**: yashdeshpandex1@gmail.com

---

## License:

This project is built with open-source components:
- Individual library licenses apply (see requirements.txt)
- Data from OpenSky Network (CC BY 4.0)
- Weather data from Open-Meteo (CC BY 4.0)

---

## Acknowlegements:

- OpenSky Network for accessible flight data
- Open-Meteo for free weather APIs
- PyTorch & Flask communities
- All open-source contributors

---

**Last Updated**: July 2026

