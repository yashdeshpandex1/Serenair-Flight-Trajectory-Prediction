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
**[15/5/2026]**: Added a chunking mechanism because RAM wasn't as powerful to input all the data at once, added a tqdm bar as well. [exams finally over]
