# download official image
FROM python:3.7

# install dependencies for running service
RUN pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN python3 -m nltk.downloader averaged_perceptron_tagger
RUN python3 -m nltk.downloader stopwords

COPY ./app /app
WORKDIR /app/

EXPOSE 8000
CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8000", "main:app", "--reload"]