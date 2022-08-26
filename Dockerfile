FROM public.ecr.aws/lambda/nodejs:16
COPY js/src/index.js ${LAMBDA_TASK_ROOT}/src/index.js
COPY package.json package-lock.json pipeline_titanic.onnx schema.json ${LAMBDA_TASK_ROOT}/
#COPY package.json  .
#COPY package-lock.json  .
#COPY pipeline_titanic.onnx .
#COPY schema.json .
RUN  npm ci
CMD [ "src/index.handler" ] 
