FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app
WORKDIR /app

# Dictionary flag lookup uses the NLTK words corpus
RUN python3 -c "import nltk; nltk.download('words')"

EXPOSE 8080
CMD ["python3", "main", "--mode", "run", "--protocol", "http"]

ENV TZ=US/Michigan
