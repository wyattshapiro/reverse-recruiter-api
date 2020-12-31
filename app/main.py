from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field
import classifier.util as classifier_util

# download model
VECTORIZER, CLASSIFIER = classifier_util.download_model()

class Email(BaseModel):
    message: str = Field(description="The body of the email")


class ClassifierResults(BaseModel):
    is_recruiter: bool = Field(description="If email is classified as a recruiter")

tags_metadata = [
    {
        "name": "classify",
        "description": "Operations to classify recruiter email.",
    },
]

app = FastAPI(
    title="ReverseRecruiter",
    description="API to detect recruiter emails",
    version="0.1.0",
    openapi_tags=tags_metadata
)


@app.post("/classify", response_model=ClassifierResults, tags=["classify"])
def classify(email: Email):
    # check if model is loaded
    if VECTORIZER and CLASSIFIER:    
        # predict if message is from a recruiter
        results = {}
        results['is_recruiter'] = classifier_util.is_recruiter(email.message, VECTORIZER, CLASSIFIER)
        return ClassifierResults(**results)