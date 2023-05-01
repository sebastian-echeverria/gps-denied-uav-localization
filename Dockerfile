FROM python:3.8

# Trusted host configs used to avoid issues when running behind SSL proxies.
RUN pip config set global.trusted-host "pypi.org pypi.python.org files.pythonhosted.org"

# Install required gdal dependencies.
RUN apt-get update
RUN apt-get install -y libgl1 libpq-dev gdal-bin libgdal-dev

# Dependencies.
WORKDIR /app/
COPY requirements.txt /app/
COPY pip_setup.sh /app/
RUN bash pip_setup.sh

# Actual code.
COPY *.sh /app/
COPY *.py /app/

ENTRYPOINT ["/bin/bash"]
