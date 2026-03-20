.PHONY: train test lint infer gradcam export detect demo docker-build docker-test clean

train:
	python main.py

test:
	python -m pytest tests/ -v

lint:
	flake8 src/ tests/ scripts/ main.py --max-line-length 100

infer:
	@test -n "$(IMG)" || (echo "Usage: make infer IMG=path/to/image.png" && exit 1)
	python scripts/infer.py $(IMG)

gradcam:
	@test -n "$(IMG)" || (echo "Usage: make gradcam IMG=path/to/image.png" && exit 1)
	python scripts/gradcam_viz.py $(IMG)

export:
	python scripts/export_onnx.py

detect:
	@test -n "$(IMG)" || (echo "Usage: make detect IMG=path/to/scene.jpg" && exit 1)
	python scripts/detect_and_classify.py $(IMG) --save

demo:
	python scripts/demo_webcam.py

docker-build:
	docker build -t traffic-sign-recognition .

docker-test:
	docker run --rm traffic-sign-recognition

clean:
	rm -rf checkpoints/ outputs/ exports/ __pycache__ src/__pycache__ tests/__pycache__
	find . -name "*.pyc" -delete
	find . -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null || true
