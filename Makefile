docker-build:
	docker build -t local-build --build-arg PYTHON_VERSION=3.6 .

docker-run-dnn:
	docker run --rm local-build:latest bash -i -c "python examples/simple_dnn.py"

docker-run-autoencoder:
	docker run --rm local-build:latest bash -i -c "python examples/autoencoder_example.py"

docker-run-cnn:
	docker run --rm local-build:latest bash -i -c "python examples/cnn_example.py"

docker-run-test:
	docker run --rm local-build:latest bash -i -c "python tests/dl_runner.py"

docker-compose-dnn:
	docker-compose --file ./docker-compose.yml bash -i -c "python examples/simple_dnn.py"
