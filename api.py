# Dependencies
from flask import Flask, request
from flask_restx import Api, Resource, fields
import traceback
import classifier.util as classifier_util


app = Flask(__name__)
api = Api(app, version='1.0', title='ReverseRecruiter API',
    description='A classifier to identify recruiter messages',
)

ns = api.namespace('reverseRecruiter')


@ns.route('/predict')
@ns.doc(params={'message': {'in': 'query', 'type': 'string'}})
class Predict(Resource):
    
    def post(self):
        # check if models are loaded on server
        # if chosen_vectorizer and chosen_classifier:
        try:
            # parse request
            message = str(request.args.get('message'))
            
            # predict if message is from a recruiter
            prediction = classifier_util.is_recruiter(message)

            return {'is_success': True, 'error': None, 'is_recruiter': str(prediction)}

        except:
            return {'is_success': False, 'error': traceback.format_exc(), 'prediction': None}
        # else:
        #     return {'is_success': False, 'error': 'Model not loaded', 'prediction': None}



if __name__ == '__main__':
    # download model
    classifier_util.download_model()

    # run flask app
    app.run(port=5000, debug=True)