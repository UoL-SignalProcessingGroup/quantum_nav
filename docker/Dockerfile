# Docker file for Quantum Navigation Toolbox
#   Note: You must disable any figure generation due to Docker limitations.
#         Currently the software does not support passing figures through
#         to the host's web-browser.

FROM python:3.12-slim

# Labels to help orginise image
LABEL name="QNavToolbox"
LABEL version="1.1.0"

# Set the working directory in the container.
WORKDIR /app

# Copy the current directory contents into the container at /app.
COPY . /app

# Install any needed dependencies specified in requirements.txt.
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8080 available to the world outside this container.
EXPOSE 8080

# Define environment variable.
ENV NAME=QNav

# Run rum.py when the container launches.
# (Swap out last argument for local path of configuration file)
CMD ["python", "run.py", "default_settings.ini"]
