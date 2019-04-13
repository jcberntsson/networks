.PHONY: env
env:
	conda activate tf

.PHONY: run-heart
run-heart: heart-main.py
	python heart-main.py

.PHONY: run-fifa
run-fifa: fifa-main.py
	python fifa-main.py
