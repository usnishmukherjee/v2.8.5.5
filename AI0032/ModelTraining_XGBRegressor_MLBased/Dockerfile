FROM python:3.8-slim-buster
LABEL "usecase"="AI0032"
LABEL "usecase_test"="AI0032_test"
COPY  requirements.txt requirements.txt
COPY  config.json config.json
COPY  aionCode.py aionCode.py
COPY  utility.py utility.py
RUN pip install --no-cache-dir -r requirements.txt