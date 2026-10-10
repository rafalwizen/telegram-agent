.PHONY: run test lint format docker-up docker-logs

run:
	python -m tgagent

test:
	pytest -q

lint:
	ruff check src tests && ruff format --check src tests

format:
	ruff format src tests && ruff check --fix src tests

docker-up:
	docker compose up -d --build

docker-logs:
	docker compose logs -f
