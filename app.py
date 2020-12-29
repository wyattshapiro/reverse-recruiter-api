# Dependencies
from flask import Flask, request
from flask_restx import Api, Resource, fields
import traceback
import classifier.util as classifier_util

# download model
VECTORIZER, CLASSIFIER = classifier_util.download_model()

app = Flask(__name__)
api = Api(app, version='1.0', title='ReverseRecruiter API',
    description='A CLASSIFIER to identify recruiter messages',
)

ns = api.namespace('reverseRecruiter')


@ns.route('/predict')
@ns.doc(params={'message': {'in': 'query', 'type': 'string'}})
class Predict(Resource):
    
    def post(self):
        # check if model is loaded
        if VECTORIZER and CLASSIFIER:
            try:
                # parse request
                message = str(request.args.get('message'))
                
                # predict if message is from a recruiter
                prediction = classifier_util.is_recruiter(message, VECTORIZER, CLASSIFIER)

                return {'is_success': True, 'error': None, 'is_recruiter': str(prediction)}

            except:
                return {'is_success': False, 'error': traceback.format_exc(), 'is_recruiter': None}
        else:
            return {'is_success': False, 'error': 'Model not loaded', 'is_recruiter': None}


if __name__ == '__main__':
    # run flask app
    app.run(port=5000, debug=True)