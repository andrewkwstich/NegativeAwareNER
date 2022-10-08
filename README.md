# NegativeAwareNER
Developing a negation-aware named entity recognition system in the product search context.

This system was created for a group UBC MDS-CL Capstone project in collaboration with Heyday by Hootsuite. The code used in the final product is found in the `code` folder (not inside `intermediate`). Of the files in `code`, I (Andrew) was mainly responsible for the following four:

- NER_Model_postprocessing.py
- Negation_Analyzer.py
- generating_dpt_store_centroids.ipynb
- new_crf_functions.py

## Steps to setup the project

- Run command `conda env update --file env.yaml --prune` to download all the required dependencies
- Create training set for NER model with command `python code/TrainSet_Generator.py`
- Run the API server with `uvicorn main:app --reload --app-dir=code/`
- Create knowledge base for product names by uploading the file using - `http://127.0.0.1:8000/uploadfile/` endpoint

_REQUEST_: 

```
curl -X 'POST' \
  'http://127.0.0.1:8000/extractInformation/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "message": "I want a blackk dress with no pink ribbons",
  "language": "English",
  "correct_spellings": true,
  "negationStyle": "tags"
}'
```
_RESPONSE_:
```
{
  "tokens": [
    "I",
    "want",
    "a",
    "black",
    "dress",
    "with",
    "no",
    "pink",
    "ribbons"
  ],
  "tags": [
    "O",
    "O",
    "O",
    "I-COLOUR",
    "I-ATTRIBUTE",
    "O",
    "O",
    "I-ATTRIBUTE",
    "I-ATTRIBUTE"
  ],
  "isNegative": [
    false,
    false,
    false,
    false,
    false,
    false,
    false,
    true,
    true
  ]
}
```
