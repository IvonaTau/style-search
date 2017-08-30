FROM ivonatau/opencv-contrib-python3.5:latest 

MAINTAINER Ivona Tautkute "ivona.tautkute@tooploox.com"

ARG THEANO_VERSION=rel-0.8.2
ARG TENSORFLOW_VERSION=1.0.0
ARG TENSORFLOW_ARCH=cpu
ARG KERAS_VERSION=1.2.2

RUN apt-get update && apt-get install -y build-essential \
    cmake \
    wget \
    unzip 

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt


# Install Theano and set up Theano config (.theanorc) 
RUN pip --no-cache-dir install git+git://github.com/Theano/Theano.git@${THEANO_VERSION} 

# Install Keras
RUN pip --no-cache-dir install git+git://github.com/fchollet/keras.git@${KERAS_VERSION}

# Install YOLO 9000
RUN wget https://github.com/IvonaTau/yolo/archive/master.zip \
	&& unzip master.zip \
	&& cd yolo-master \
	&& make clean \
	&& make all\
	&& cp libdarknetlnx.so ../libdarknetlnx.so \
	&& wget https://www.dropbox.com/s/5tckwl995ksf51h/yolo.weights?dl=1 \
	&& cp yolo.weights?dl=1 ../yolo.weights

COPY . /app


COPY keras.json /root/.keras/keras.json

ENV KERAS_BACKEND=theano

RUN python3 copy_vgg_weights.py

EXPOSE 3000

CMD python3 run.py