#!/bin/bash

set -e

build() {
  echo "Building the Docker image..."
  docker build -t fastapi-gradcam .
  echo "Build complete."
}

run() {
  echo "Running the Docker container..."
  docker run -d -p 8000:8000 --name fastapi-gradcam-container fastapi-gradcam
  echo "Container is running and accessible at http://localhost:8000"
}

stop() {
  echo "Stopping the Docker container..."
  docker stop fastapi-gradcam-container || echo "No running container to stop."
  docker rm fastapi-gradcam-container || echo "No container to remove."
  echo "Stopped and removed the container."
}

logs() {
  echo "Showing logs of the Docker container..."
  docker logs fastapi-gradcam-container
}

test() {
  echo "Running unit tests..."
  pytest --disable-warnings
}

test_coverage() {
  echo "Running unit tests with coverage..."
  pytest --cov=app tests/
}

rebuild() {
  echo "Rebuilding the Docker image and container..."
  stop
  build
  run
}

help() {
  echo "Usage: ./scripts.sh [command]"
  echo "Commands:"
  echo "  build           Build the Docker image"
  echo "  run             Run the Docker container"
  echo "  stop            Stop the running Docker container"
  echo "  logs            Show logs of the running Docker container"
  echo "  test            Run unit tests"
  echo "  test_coverage   Run unit tests with coverage report"
  echo "  rebuild         Rebuild the Docker image and restart the container"
  echo "  help            Show this help message"
}

case "$1" in
  build) build ;;
  run) run ;;
  stop) stop ;;
  logs) logs ;;
  test) test ;;
  test_coverage) test_coverage ;;
  rebuild) rebuild ;;
  help | *) help ;;
esac
