FROM svizor/zoomcamp-model:3.10.12-slim

RUN pip install pipenv

RUN pwd

COPY Pipfile ./
COPY Pipfile.lock ./

RUN pipenv install

COPY predict.py ./

RUN ls

EXPOSE 8823

ENTRYPOINT ["pipenv", "run", "gunicorn", "--bind=0.0.0.0:8823", "predict:app"]