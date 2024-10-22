FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Install dependencies

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . /app

ENTRYPOINT ["python", "main.py"]
