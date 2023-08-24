build:
	qwak models build . --model-id cancer_detection --main-dir main

deploy:
	qwak models deploy realtime --model-id cancer_detection --pods 1

build-features:
	qwak features register -p main/feature_store.py

