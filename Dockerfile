# start from official image
FROM python:3.7
WORKDIR /code
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# install dependencies for running service
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN python3 -m nltk.downloader averaged_perceptron_tagger
RUN python3 -m nltk.downloader stopwords

EXPOSE 5000
COPY . .
CMD ["flask", "run"]