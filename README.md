# Serenair-Flight-Trajectory-Detection

**Real-time Flight Trajectory Prediction with Machine Learning Models**

Serenair is a full-stack machine learning application that harnesses the power of LSTM neural networks to predict aircraft position at next-instance and 10~ minutes ahead in real time using live ADS-B and wind data.
Furthermore, it makes use of DBSCAN clustering algorithm on these predictions to identify aircraft density/traffic clusters 10~ minutes ahead in time.


**www.serenair.live**

---

## Features:
- Next Instance Prediction 
- Extended Horizon Prediction
- Traffic Density/Convergence Forecasting
- Real-time Dashboard
- Multi-Continent Support
- Weather Integration
- Interactive Maps

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

---

## Performance and Results:

---

## Working:

---

## Data Sources and Credits:

---

## Future Enhancements:

---

## Contact me:

---

## License:

---

## Acknowlegements:

---


