## Helpful links

* https://onnx.ai/sklearn-onnx/auto_examples/plot_complex_pipeline.html
* http://onnx.ai/sklearn-onnx/pipeline.html
* https://onnx.ai/sklearn-onnx/auto_examples/plot_pipeline_xgboost.html

## Training a model

With onnx, just easier to use conda...at least on a mac

`conda create -n .venv python=3.9.2` 
`conda init zsh`
`conda install --file requirements.txt`


`conda install skl2onnx onnxruntime scikit-learn onnx pydot matplotlib`

`conda run train_model.py`

## Invoking the model

`npm ci`
`npm run start`

## building/transpiling to JS

`npm run build`


## testing lambda

`docker build . -t testonnx`

`docker run -p 9000:8080 testonnx`

In a new terminal,

`curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -d "{\"body\":[{\"sex\":\"female\", \"age\":20.0, \"fare\":3.5, \"embarked\":\"S\", \"pclass\":1}]}"`