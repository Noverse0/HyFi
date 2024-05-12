default: build

help:
	@echo 'Management commands for hyfi:'
	@echo
	@echo 'Usage:'
	@echo '    make build            Build the hyfi project.'
	@echo '    make pip-sync         Pip sync.'

preprocess:
	@docker 

build:
	@echo "Building Docker image"
	@docker build . -t hyfi 

run:
	@echo "Booting up Docker Container"
	@docker run -it --gpus '"device=0"' --ipc=host --name hyfi -v `pwd`:/workspace/hyfi hyfi:latest /bin/bash

up: build run

rm: 
	@docker rm hyfi

stop:
	@docker stop hyfi

reset: stop rm