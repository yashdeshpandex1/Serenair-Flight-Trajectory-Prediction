# Serenair-Flight-Trajectory-Detection
Retrieves data from OpenSky api to predict flight trajectory.
<br></br>

> You can get a copy of the raw global dataset that I collected from OpenSky here: [OpenSky_raw_dataset](https://github.com/yashdeshpandex1/Serenair-Flight-Trajectory-Detection/releases/download/v1.0/opensky_raw.csv)
<br></br>

### Database Schema:
- Created two tables
  1. aircraft - has information about the aircrafts (icao24 and origin country).
  2. aircraft states - has all the aircraft state vectors.
 
<img width="401" height="496" alt="db_schema" src="https://github.com/user-attachments/assets/783daa08-1a09-4c06-98f6-efec5d918b44" />

<br></br>

### Deployment:
- I created a postgreSQL database on Azure.
  configuration: Burstable, B1ms, 1 vCores, 2 GiB RAM, 32 GiB storage. I found this low-end resource ideal for this project.
- Mlflow server used for Experiment tracking is setup using Azure MLflow and uses a postgreSQL database.
