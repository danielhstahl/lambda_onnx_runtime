FROM public.ecr.aws/lambda/nodejs:16
COPY js/src/index.js ${LAMBDA_TASK_ROOT}/src/index.js
ENV NODE_ENV production
COPY package.json package-lock.json ${LAMBDA_TASK_ROOT}/
RUN  npm ci --production
### TODO, split this into a base image which contains everything except the onnx and the schema
### Then each instantiation of a model will build from the base and add schema.json and the onnx
COPY pipeline_titanic.onnx schema.json ${LAMBDA_TASK_ROOT}/
CMD [ "src/index.handler" ] 
