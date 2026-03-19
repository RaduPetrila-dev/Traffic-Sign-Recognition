.PHONY: train test infer lint clean

train:
	python main.py

test:
	python -m pytest tests/ -v

infer:
	@echo "Usage: make infer IMG=path/to/image.png"
	@test -n "$(IMG)" && python scripts/infer.py $(IMG) || echo "Set IMG=path/to/image.png"

lint:
	python -m flake8 src/ tests/ scripts/ main.py --max-line-length 100

clean:
	rm -rf checkpoints/ outputs/ __pycache__ src/__pycache__ tests/__pycache__
	find . -name "*.pyc" -delete
