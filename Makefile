.PHONY: numba-clean numba-cache-clean python-cache-clean cache-clean safety_net
safety_net:
	@echo No target specified, exiting.

numba-cache-clean:
	find -type f -name "*.nb[ci]" -delete
	@echo Done cleaning numba cache files.

python-cache-clean:
	find -type f -name "*.py[co]" -delete
	find -type d -name "__pycache__" -delete
	@echo Done cleaning pycache.

cache-clean: numba-cache-clean python-cache-clean
	@echo Done.
