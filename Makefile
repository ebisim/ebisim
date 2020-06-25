.PHONY: numba-clean safety_net
safety_net:
	@echo No target specified, exiting.

numba-clean:
	find -type f -name "*.nb[ci]" -delete