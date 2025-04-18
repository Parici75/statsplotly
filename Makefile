PROJECT_NAME=statsplotly

# Python env
PYTHON_SHELL_VERSION := $(shell python --version | cut -d " " -f 2)
POETRY_AVAILABLE := $(shell which poetry > /dev/null && echo 1 || echo 0)

# CI variables
CI_EXCLUDED_DIRS = __pycache__ dist tests docs
CI_DIRECTORIES=$(filter-out $(CI_EXCLUDED_DIRS), $(foreach dir, $(dir $(wildcard */)), $(dir:/=)))

# Project targets
confirm:
	@echo "Are you sure? [y/N] " && read ans && [ $${ans:-N} = y ]

exists-%:
	@which "$*" > /dev/null && echo 1 || echo 0

print-%: ; @echo $* = $($*)

init:
ifneq ($(POETRY_AVAILABLE), 1)
	@echo "No Poetry executable found, cannot init project" && exit 1;
endif
	@poetry check --no-ansi --quiet
	@echo "✅ Poetry is installed"
	@echo "💡 Using Python $(PYTHON_SHELL_VERSION)"
	@poetry config virtualenvs.in-project true
	@poetry config virtualenvs.create true
	@poetry install

	pre-commit

# CI targets
lint-%:
	@echo lint-"$*"
	@poetry run black --check "$*"
	@poetry run isort --check "$*"
	@poetry run ruff check "$*"
	@echo "    ✅ All good"

lint: $(addprefix lint-, $(CI_DIRECTORIES))

typecheck-%:
	@echo typecheck-"$*"
	@poetry run mypy "$*"

typecheck: $(addprefix typecheck-, $(CI_DIRECTORIES))

test:
	@poetry run pytest -s --rootdir ./  --cache-clear tests

ci: lint typecheck test

coverage:
	@poetry run coverage run -m pytest
	@poetry run coverage report

# Pre-commit hooks
set-pre-commit:
	@echo "Setting up pre-commit hooks..."
	@poetry run pre-commit install
	@poetry run pre-commit autoupdate

run-pre-commit:
	@poetry run pre-commit run --all-files


# Documentation
update-doc:
	@poetry run sphinx-apidoc --module-first --no-toc --force -o docs/source $(PROJECT_NAME)

build-doc:
	@poetry run sphinx-build docs ./docs/_build/html/


# Cleaning
clean-python:
	@echo "🧹 Cleaning Python bytecode..."
	@poetry run pyclean . --quiet

clean-cache:
	@echo "🧹 Cleaning cache..."
	@find . -regex ".*_cache" -type d -print0|xargs -0 rm -r --
	@poetry run pre-commit clean

clean-hooks:
	@echo "🧹 Cleaning hooks..."
	@rm -r ".git/hooks" ||:


# Global
clean: confirm clean-cache clean-python clean-hooks
	@echo "✨ All clean"
