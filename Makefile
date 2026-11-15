.PHONY: style quality check-keywords

# make sure to test the local checkout in scripts and not the pre-installed one (don't use quotes!)
export PYTHONPATH = open_instruct

check_dirs := open_instruct *mason.py

style:
	uv run ruff format $(check_dirs)

quality: check-keywords
	uv run ruff check -q --fix $(check_dirs)
	uv run python -m compileall -qq $(check_dirs)
	uv run ty check

style-check:   ## *fail* if anything needs rewriting
	uv run ruff format --check --diff $(check_dirs)

quality-check: check-keywords ## *fail* if any rewrite was needed
	uv run ruff check --exit-non-zero-on-fix $(check_dirs)
	uv run ty check
	uv run python -m compileall -qq $(check_dirs)

# TYPE_CHECKING is allowed so optional dependencies stay out of runtime imports.
# Share the remaining keyword policy between pre-commit and CI.
check-keywords:
	@test -d open_instruct/
	@grep -rn --include='*.py' -E 'nonlocal[[:space:]]' open_instruct/; test $$? -eq 1

# Portable checks intentionally bypass the root CUDA dependency resolver.
MAC_PYTHON ?= .venv-mac/bin/python
mac_dirs := open_instruct/portable tests/portable scripts/portable

.PHONY: mac-setup mac-test mac-check mac-demo
mac-setup:
	bash scripts/portable/setup.sh

mac-test:
	PYTHONPATH=. $(MAC_PYTHON) -m pytest tests/portable -q

mac-check:
	$(MAC_PYTHON) -m ruff check $(mac_dirs)
	$(MAC_PYTHON) -m ruff format --check $(mac_dirs)
	$(MAC_PYTHON) -m compileall -q open_instruct/portable
	$(MAKE) mac-test

mac-demo:
	PYTHONPATH=. $(MAC_PYTHON) -m open_instruct.portable run --config configs/portable/tiny.json --output output/mac-demo
