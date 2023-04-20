FROM python:3.8

# Trusted host configs used to avoid issues when running behind SSL proxies.
RUN pip config set global.trusted-host "pypi.org pypi.python.org files.pythonhosted.org"

# Dependencies.
WORKDIR /app/
COPY requirements.txt /app/
RUN pip install -r requirements.txt --default-timeout=100

# Actual code.
COPY *.sh /app/
COPY *.py /app/

ENTRYPOINT ["/bin/bash"]
