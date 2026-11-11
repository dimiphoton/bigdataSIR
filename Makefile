.PHONY: install test lint simulate fit report app play

install:
	uv sync --all-groups

test:
	uv run pytest

lint:
	uv run ruff check src/epidemic app tests
	uv run ruff format --check src/epidemic app tests
	uv run mypy src/epidemic

simulate:
	uv run epidemic simulate

play:
	uv run epidemic play

fit:
	uv run epidemic fit

report:
	uv run epidemic report

app:
	uv run streamlit run app/accueil.py
