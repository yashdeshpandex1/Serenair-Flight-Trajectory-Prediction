## Project Timeline
**[3/4/2026]**: Created the repository, defined the structure of repo, built a docker image for preprocessing directory as well as a basic docker container. 
            Furthermore, I also designed a data collection script to collect global flights from OpenSky api and learnt about its rate limits.
            <br></br>
**[4/4/2026]**: Finally, collected the data from opensky api with 1,000,000 rows and a bunch of features.
                <br></br>
**[6/4/2026]**: Introduced utilities, data cleaning and feature engineering scripts.
            <br></br>
**[1/5/2026]**: Wrote a connection script to Azure PostgreSQL database and updated docker container. [delay because I was busy with college commitments]
            <br></br>
**[2/5/2026]**: Created a database schema having two tables - aircraft and aircraft states.
            <br></br>
**[15/5/2026]**: Added a chunking mechanism because RAM wasn't as powerful to input all the data at once, added a tqdm bar as well. 
            <br></br>
**[20/5/2026]**: I added a 'prepare training data' script that makes use of all the preprocessing and engineering scripts I created earlier. 
            It makes data appropriate for using model on it. [p.s: exams finally over!]
            I also created scripts for creating sequences of data for RNN and saving it in npz format.
            <br></br>
**[21/5/2026]**: I made changes to training data prep scripts I made earlier as well I fixed issue where Pytorch would fail to import.
            <br></br>
**[22/5/2026]**: I setup mlflow tracking server for experiment tracking using Azure ML. 
            <br></br>
**[23/5/2026]**: Verified and fixed all my preprocessing scripts. Replaced Delta lat and long with absolute versions of them in the features.
            <br></br>
**[24/5/2026]**: Created a dataloaders script for model training.
            <br></br>
