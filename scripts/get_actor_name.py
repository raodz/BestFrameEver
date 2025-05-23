import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "..//bestframeever" "-KEY.json"

import vertexai
from vertexai.preview.generative_models import GenerativeModel

PROJECT_ID = "bestframeever"
REGION = "us-central1"

vertexai.init(project=PROJECT_ID, location=REGION)

model = GenerativeModel("gemini-2.0-flash-001")

movie = "Django"

response = model.generate_content(
    "Give the names of the actors or actresses "
    "who played the three main roles in the "
    f"movie {movie}. Be as concise as possible. "
    "Make sure you name the correct actors in "
    "the correct order. Give your answer in the "
    "form: First Name Last Name, First Name "
    "Last Name, First Name Last Name"
)
print(response.text)
