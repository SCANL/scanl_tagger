FROM python:3.12-slim

# Install (and build) requirements
COPY requirements.txt /requirements.txt
RUN apt-get update && \
    apt-get install -y git curl && \
    pip install -r requirements.txt && \
    rm -rf /var/lib/apt/lists/*

# ntlk downloads
RUN python3 -c "import nltk; nltk.download('averaged_perceptron_tagger');nltk.download('universal_tagset')"

# Pythong scripts and data
COPY classifier_multiclass.py \
     download_code2vec_vectors.py \
     feature_generator.py \
     tag_identifier.py \
     create_models.py \
     version.py \
     serve.json \
     main \
     /.
COPY input/* /input/.
COPY models/model_GradientBoostingClassifier.pkl /models/.

CMD date; \
    echo "Download..."; \
    remote_target_date=$(curl -sI http://131.123.42.41/target_vecs.txt | grep -i "Last-Modified" | cut -d' ' -f2-); \
    remote_token_date=$(curl -sI http://131.123.42.41/token_vecs.txt | grep -i "Last-Modified" | cut -d' ' -f2-); \
    remote_words_date=$(curl -sI http://131.123.42.41/abbreviationList.csv | grep -i "Last-Modified" | cut -d' ' -f2-); \
    remote_dictionary_date=$(curl -sI htp://131.123.42.41/en.txt | grep -i "Last-Modified" | cut -d' ' -f2-); \
    if [ -n "$remote_target_date" ] && [ -n "$remote_token_date" ]; then \
        remote_target_timestamp=$(date -d "$remote_target_date" +%s); \
        remote_token_timestamp=$(date -d "$remote_token_date" +%s); \
        remote_words_timestamp=$(date -d "$remote_words_date" +%s); \
        remote_dictionary_timestamp=$(date -d "$remote_dictionary_date" +%s); \
        if [ ! -f /code2vec/target_vecs.txt ] || [ $remote_target_timestamp -gt $(date -r /code2vec/target_vecs.txt +%s) ]; then \
            curl -s -o /code2vec/target_vecs.txt http://131.123.42.41/target_vecs.txt; \
            echo "target_vecs.txt updated"; \
        else \
            echo "target_vecs.txt not updated"; \
        fi; \
        if [ ! -f /code2vec/token_vecs.txt ] || [ $remote_token_timestamp -gt $(date -r /code2vec/token_vecs.txt +%s) ]; then \
            curl -s -o /code2vec/token_vecs.txt http://131.123.42.41/token_vecs.txt; \
            echo "token_vecs.txt updated"; \
        else \
            echo "token_vecs.txt not updated"; \
        fi; \
        if [ ! -r /words/abbreviationList.csv ] || [ $remote_words_timestamp -gt $(date -r /words/abbreviationList.csv +%s) ]; then \
            curl -s -o /words/abbreviationList.csv http://131.123.42.41/abbreviationList.csv; \
            echo "abbreviationList.csv updated"; \
        else \
            echo "abbreviationList.csv not updated"; \
        fi; \
        if [ ! -r /words/en.txt ] || [ $remote_dictionary_timestamp -gt $(date -r /words/en.txt +%s) ]; then \
            curl -s -o /words/en.txt http://131.123.42.41/en.txt; \
            echo "en.txt updated"; \
        else \
            echo "en.txt not updated"; \
        fi; \
    else \
        echo "Failed to retrieve Last-Modified headers"; \
    fi; \
    date; \
    echo "Running..."; \
    /main -r --words words/abbreviationList.csv

ENV TZ=US/Michigan
