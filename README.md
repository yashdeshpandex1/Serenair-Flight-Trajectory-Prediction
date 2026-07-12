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

### 1. Data Collection
- Collects global flight data from OpenSky API once the background worker is triggered (Background workers are triggered once anyone visits the website).
- Ingests all the data onto the PostgreSQL database, prune old data (>15 mins).
- The relevant weather/wind data is integrated at the time of data processing/prediction.

### 2. Prediction Pipeline
- The user navigates to a page and triggers "/api/wakeup" which starts the background processes.
- Background worker preprocesses real-time data:
     - Cleans old and invalid records.
     - Extracts useful engineered features.
     - Creates a ten step sliding window.
     - Separates independent/feature and dependent/target variables.
     - Scales features using existing pre-trained scalers.
- Models run inference on sequences for each continent and results are cached in Redis (prevents cold start).
- Finally, visualisation maps are rendered using Bokeh.

### 3. Dataset and Training

## Data Source
- **OpenSky Network API**: Global ADS-B flight tracking data
- **Collection Date**: April 4, 2026
- **Dataset Size**: ~1 million observations
- **Polling Frequency**: 10-second intervals
- **Coverage**: Global regions

## Features (Total: 21)
 
### ADS-B Features (8)
- Position: latitude, longitude, barometric altitude
- Kinematics: velocity, vertical rate, true track
- Time: timestamp, aircraft category
### Engineered Features (6)
- Temporal: delta_time, hour_sin, hour_cos
- Kinematic: acceleration, turn_rate, climb_phase
### Meteorological Features (3)
- Wind speed at 250 hPa (jet stream altitude)
- Wind direction at 250 hPa
- Temperature at 2 meters (surface level)
### Target Variables (2)
- Δ Latitude (change in latitude)
- Δ Longitude (change in longitude)
## Data Preprocessing
- **Scaling**: StandardScaler for features, RobustScaler for targets
- **Sequence Creation**: 10-timestep sliding window
- **Train/Test Split**: Group Shuffle Split (70/15/15) at flight level
- **Validation**: Stratified by aircraft ICAO24 identifier


---

## Data Sources and Credits:

---

## Future Enhancements:

---

## Contact me:
- **LinkedIn**: [Your LinkedIn]
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

