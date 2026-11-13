FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 TF_CPP_MIN_LOG_LEVEL=2
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt && useradd -m app
COPY . .
RUN chown -R app /app
USER app
EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s CMD python -c "import urllib.request as u;u.urlopen('http://localhost:5000/health')"
CMD ["gunicorn","-b","0.0.0.0:5000","-w","1","--threads","4","--timeout","120","app:app"]
