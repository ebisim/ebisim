FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY dashboard.py ./
COPY ebisim ./ebisim

# CMD ["python", "./dashboard.py"]
CMD ["gunicorn", "-b", "0.0.0.0:8050", "-w", "4", "dashboard:server"]