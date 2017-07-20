FROM java:8
ENV workdir /code
ENV maven_bin /apache-maven-3.5.0-bin.zip
WORKDIR ${workdir}

RUN apt-get update && true

# Install maven
ADD http://mirror.netcologne.de/apache.org/maven/maven-3/3.5.0/binaries/apache-maven-3.5.0-bin.zip ${maven_bin}
RUN cd / && unzip ${maven_bin}
RUN ln -s /apache-maven-3.5.0/bin/mvn /usr/sbin/mvn
RUN ln -s /apache-maven-3.5.0/bin/mvnDebug /usr/sbin/mvnDebug
RUN ln -s /apache-maven-3.5.0/bin/mvnyjp /usr/sbin/mvnyjp

# Install sbt
RUN echo "deb https://dl.bintray.com/sbt/debian /" | tee -a /etc/apt/sources.list.d/sbt.list
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 2EE0EA64E40A89B84B2DF73499E82A75642AC823
RUN apt-get install -y apt-transport-https && apt-get update && apt-get install -y sbt

# Install openie
RUN git clone https://github.com/knowitall/openie /openie
RUN cd /openie && sbt -J-Xmx2500M compile
RUN cd /openie && sbt -J-Xmx2000M assembly

#RUN mvn install