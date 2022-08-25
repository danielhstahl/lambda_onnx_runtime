FROM public.ecr.aws/lambda/nodejs:16
COPY *.js ${LAMBDA_TASK_ROOT}/
COPY package.json  .
COPY package-lock.json  .
COPY pipeline_titanic.onnx .
COPY schema.json .
RUN  npm ci
CMD [ "index.main" ] 
