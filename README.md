# Sales_prediction_ETL

## Overview

## Building and Running with Docker

### Building the Docker Image

Use the `docker build` command to build your Docker image from the Dockerfile. This command creates an image according to the instructions in the Dockerfile.
It contains necessary libraries like Pyspark, Pandas, Scikitlearn, PostgresQL, etc

```bash
docker build -t your-image-name .
```

### Running the container

```bash
docker-compose up -d
```

### NOte: Place the necessary files in the dag directory of airflow after building image. It is mounted to the docker container.
