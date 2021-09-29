BASE_IMAGE_NAME=gslsquantification
# Extend the space-separated list of your Python modules/packages in
# the root directory of this project to install them (with their
# dependencies) for use in your notebooks, and to run other commands
# on them (like linting and unit tests).
PYTHON_MODULES=pyquantification

# Building and dependencies
env:
	echo "BASE_IMAGE_NAME=${BASE_IMAGE_NAME}" > .env
build: env
	docker-compose build \
		--build-arg GROUP_ID=`id -g` \
		--build-arg USER_ID=`id -u`
deps: build
	for module in $(PYTHON_MODULES); do \
		docker-compose run --rm  --workdir="/home/jovyan/work" jupyter \
			pip install --user -e "$${module}[dev]" ; \
	done
clear-build:
	docker-compose rm
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml rm

# Running the application
run: deps
	docker-compose up
run-prod:
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
stop:
	docker-compose stop

# Starting a shell in a Docker container
bash:
	docker-compose exec jupyter /bin/bash
sudo-bash:
	docker-compose exec --user root jupyter /bin/bash
run-bash:
	docker-compose run --rm jupyter /bin/bash
run-sudo-bash:
	docker-compose run --user root --rm jupyter /bin/bash

# Python module utilities
lint:
	for module in $(PYTHON_MODULES); do \
		docker-compose run --rm --workdir="/home/jovyan/work" jupyter \
			flake8 "$${module}"; \
	done
test:
	for module in $(PYTHON_MODULES); do \
		docker-compose run --rm --workdir="/home/jovyan/work/$${module}" jupyter \
			pytest \
			--cov="$${module}" \
			--cov-report="html:test/coverage" \
			--cov-report=term ; \
	done
types:
	for module in $(PYTHON_MODULES); do \
		docker-compose run --rm --workdir="/home/jovyan/work/$${module}" jupyter \
			mypy . ;\
	done
