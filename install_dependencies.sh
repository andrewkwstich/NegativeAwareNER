
conda env update --file env.yaml --prune
conda activate NER
python -m nltk.downloader averaged_perceptron_tagger
python -m spacy download en_core_web_sm
