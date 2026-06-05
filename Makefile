format:
	uv run ruff format

lint:
	uv run ruff check

type:
	uv run ty check

run:
	uv run learn2slither 

flake:
	uv run flake8 src