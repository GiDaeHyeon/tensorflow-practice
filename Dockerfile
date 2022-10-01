FROM tensorflow/tensorflow:2.6.1-gpu

ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get -y install python3-pip

WORKDIR /home/tensorflow-practice