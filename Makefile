chunk:
	python3 chunks/chunker.py

test_chunk:
	python3 test_chunker.py

requirements:
	pip install -r requirements.txt

index:
	python -m scripts.build_index

run:
	python -m rag.rag_main

ui:
	streamlit run streamlit_app.py