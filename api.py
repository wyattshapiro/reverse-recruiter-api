# Dependencies
from flask import Flask, request
from flask_restx import Api, Resource, fields
import pickle
import traceback

CHOSEN_MODEL = 'RidgeClassifier-TfidfVectorizer'
CHOSEN_VECTORIZER_PATH = 'classifier/models/{}/{}'.format(CHOSEN_MODEL, 'vectorizer.pkl')
CHOSEN_CLASSIFIER_PATH = 'classifier/models/{}/{}'.format(CHOSEN_MODEL, 'classifier.pkl')

app = Flask(__name__)
api = Api(app, version='1.0', title='ReverseRecruiter API',
    description='A classifier to identify recruiter messages',
)

ns = api.namespace('reverseRecruiter')


@ns.route('/predict')
class Predict(Resource):
    
    def post(self):
        if chosen_vectorizer and chosen_classifier:
            try:
                return {'prediction': str('0')}

            except:
                return {'trace': traceback.format_exc()}
        else:
            print ('Train the model first')
            return ('No model here to use')



if __name__ == '__main__':
    try:
        port = 5000
    except:
        port = 12345

    # read in model
    chosen_vectorizer = pickle.load(open(CHOSEN_VECTORIZER_PATH, "rb"))
    chosen_classifier = pickle.load(open(CHOSEN_CLASSIFIER_PATH, "rb"))

    # run app
    app.run(port=port, debug=True)