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

## building TS

`npm run build`


## testing lambda

`docker run -d -v ~/.aws-lambda-rie:/aws-lambda -p 9000:8080 \
  --entrypoint /aws-lambda/aws-lambda-rie hello-world:latest <image entrypoint> \
      <(optional) image command>`