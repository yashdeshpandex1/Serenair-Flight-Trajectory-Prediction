# Official Jupyter Environment
FROM jupyter/scipy-notebook:latest

# Set the working directory
WORKDIR /Serenair

# Copy the dependencies from requirements file
COPY ./preprocessing/requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt