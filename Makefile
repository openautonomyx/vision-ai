.PHONY: build

build:
	@if command -v docker >/dev/null 2>&1; then \
		echo "Docker detected: building dev image with docker compose"; \
		docker compose --profile dev build; \
	else \
		echo "Docker not found; running Python bytecode compilation fallback"; \
		python -m compileall app.py auth.py metrics.py rate_limit.py mcp_server.py; \
	fi
