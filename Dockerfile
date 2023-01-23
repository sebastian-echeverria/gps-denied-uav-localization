FROM python:3.7.9

RUN pip install pipenv

# Dependencies.
WORKDIR /app/
COPY Pipfile /app/
#RUN pipenv lock
COPY Pipfile.lock /app/
RUN pipenv install --system --deploy --ignore-pipfile

# Actual code.
COPY deep_feat/ /app/deep_feat/
COPY *.sh /app/
COPY *.py /app/

ENTRYPOINT ["/bin/bash"]
