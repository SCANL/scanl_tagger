FROM python:3.10-alpine3.19

# Install (and build) requirements
COPY requirements.txt /requirements.txt
ENV INSTALLED_PACKAGES="build-base \
    gfortran \
    pkgconf \
    py3-scipy \
    openblas-dev \
    linux-headers"
ENV OTHER_PACKAGES="libstdc++ openblas libgomp git"
RUN apk update && \
    apk upgrade && \
    apk add $INSTALLED_PACKAGES && \
    apk add     $OTHER_PACKAGES && \
    pip install -r requirements.txt
    # apk del $INSTALLED_PACKAGES

RUN pip3 install flask
RUN pip3 install git+https://github.com/cnewman/spiral.git
RUN pip3 install nltk
RUN python3 -c "import nltk; nltk.download('averaged_perceptron_tagger');nltk.download('universal_tagset')"

COPY classifier_multiclass.py \
     download_code2vec_vectors.py \
     feature_generator.py \
     print_utility_functions.py \
     tag_identifier.py \
     serve.json \
     main.py \
     /.
COPY input/det_conj_db2.db /input/.

CMD date; \
    echo "Download..."; \
    mkdir /code2vec; \
    wget -o /code2vec/target_vecs.txt http://131.123.42.41/target_vecs.txt; \
    wget -o /code2vec/token_vecs.txt http://131.123.42.41/token_vecs.txt; \
    date; \
    echo "Training..."; \
    python3 /main.py -t; \
    date; \
    echo "Running..."; \
    python3 /main.py -r

ENV TZ=US/Michigan
