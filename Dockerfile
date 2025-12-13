FROM python:3.10

# Install Playwright + Browsers
RUN pip install playwright
RUN playwright install chromium

# Install system dependencies for Playwright
RUN apt-get update && apt-get install -y \
    libnss3 \
    libatk1.0-0 \
    libcups2 \
    libxdamage1 \
    libxrandr2 \
    libasound2 \
    libatk-bridge2.0-0 \
    libgtk-3-0 \
    libxkbcommon0 \
    libdrm2 \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Copy app
WORKDIR /app
COPY ./app /app

# Install requirements
RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]
