FROM public.ecr.aws/lambda/python:3.10

RUN pip install numpy Pillow

RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl

COPY clothing_xception_v4_23_0.911.tflite .

COPY model_script.py .

CMD [ "model_script.lambda_handler"]

