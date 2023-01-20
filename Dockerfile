FROM python:3.7.9

RUN pip install pipenv

# Actual code.
COPY deep_feat/ /app/deep_feat/
COPY optimize/ /app/optimize/
#COPY Pipfile /app/
COPY requirements.txt /app/
WORKDIR /app/

# Installing Python deps without a venv (not needed in container).
#RUN pipenv lock
#RUN pipenv install --system --deploy --ignore-pipfile
RUN pipenv install -r requirements.txt

COPY *.sh /app/
COPY *.py /app/

ENTRYPOINT ["/bin/bash"]
