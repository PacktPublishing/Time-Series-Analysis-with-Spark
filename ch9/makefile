build:
	make prep && docker-compose build
	
up:
	make prep && docker-compose up -d

down:
	docker-compose down

restart:
	make down && make up

rebuild:
	make down && make build

prep:
	sh prep-airflow.sh
