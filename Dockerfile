FROM python:3.10-slim

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
     print_utility_functions.py \
     tag_identifier.py \
     serve.json \
     main \
     /.
COPY input/det_conj_db2.db /input/.

CMD date; \
    echo "Download..."; \
    remote_target_date=$(curl -sI http://131.123.42.41/target_vecs.txt | grep -i "Last-Modified" | cut -d' ' -f2-); \
    remote_token_date=$(curl -sI http://131.123.42.41/token_vecs.txt | grep -i "Last-Modified" | cut -d' ' -f2-); \
    if [ -n "$remote_target_date" ] && [ -n "$remote_token_date" ]; then \
        remote_target_timestamp=$(date -d "$remote_target_date" +%s); \
        remote_token_timestamp=$(date -d "$remote_token_date" +%s); \
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
    else \
        echo "Failed to retrieve Last-Modified headers"; \
    fi; \
    date; \
    echo "Training..."; \
    /main -t; \
    date; \
    echo "Running..."; \
    /main -r

ENV TZ=US/Michigan
