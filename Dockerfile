ARG PYTHON_VERSION=3.10.5
FROM python:${PYTHON_VERSION}-slim as base

# Update the base image
#RUN apk update && apk upgrade --no-cache

# Install build-base to enable scikit learn installation
#RUN apk add build-base


# Install pip during base image build
RUN python -m ensurepip --upgrade

# Upgrade PIP
#RUN pip3 install --upgrade pip

ENV PYTHONDONTWRITEBYTECODE=1

ENV PYTHONUNBUFFERED=1

WORKDIR /app

ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

COPY requirements.txt /app/

# Install dependencies from requirements.txt
RUN pip install -r requirements.txt

# Install pandas library
RUN pip install pandas

#RUN --mount=type=cache,target=/root/.cache/pip \
#    --mount=type=bind,source=requirements.txt,target=requirements.txt \
#    python -m pip install -r requirements.txt

#RUN pip3 install -r /requirements.txt

# Copy the source code into the container.
COPY data/ /app/data/
COPY test.py /app/
COPY train.py /app/

RUN chown -R appuser:appuser /app/

USER appuser

RUN python train.py

# Run the test scrip
CMD ["python", "test.py"]
