# Start from the latest Apache Airflow image
FROM apache/airflow:latest

# Install additional dependencies
RUN pip install pandas matplotlib scikit-learn pyspark

# Switch to root user for installing wget and Java
USER root

# Install wget and unzip
RUN apt-get update && \
    apt-get install -y wget unzip && \
    rm -rf /var/lib/apt/lists/*

# Download and install JDK
RUN wget -qO /tmp/openjdk.zip https://download.java.net/java/GA/jdk22.0.1/c7ec1332f7bb44aeba2eb341ae18aca4/8/GPL/openjdk-22.0.1_windows-x64_bin.zip && \
    unzip -q /tmp/openjdk.zip -d /usr/local && \
    rm /tmp/openjdk.zip && \
    mv /usr/local/jdk-22.0.1 /usr/local/java

# Set Java environment variables
ENV JAVA_HOME=/usr/local/java
ENV PATH=$PATH:$JAVA_HOME/bin

# Install default-jdk (which includes OpenJDK)
RUN apt-get update && \
    apt-get install -y default-jdk && \
    rm -rf /var/lib/apt/lists/*

# Download and install Apache Spark
RUN wget -qO- https://dlcdn.apache.org/spark/spark-3.5.1/spark-3.5.1-bin-hadoop3.tgz | tar xz -C /opt \
    && mv /opt/spark-3.5.1-bin-hadoop3 /opt/spark

# Set environment variables
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin

# Switch back to the original user
USER airflow

# Optionally, set the UID for Airflow processes
ENV AIRFLOW_UID=1000
