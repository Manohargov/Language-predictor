FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file into the container
COPY ./requirements.txt /app/requirements.txt

# Install the Python dependencies
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copy your application code into the container
COPY . /app/app

# Expose the port your application will run on
EXPOSE 80

# This is the correct command to run a FastAPI app with Uvicorn
# The format is: uvicorn <module_name>:<instance_name>
# In your case, it would be 'app.main:app'
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]