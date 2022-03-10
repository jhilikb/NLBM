apt update
sudo docker run -it --rm --network=host --gpus all -v `pwd`/NLBM:/home/NLBM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
cd /home/NLBM
mkdir build
cd build
mkdir bin
cd bin
cp /home/NLBM/models/* .
cp /home/NLBM/python/* .
