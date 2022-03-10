apt update
sudo docker run -it --rm --network=host --gpus all -v `pwd`:/home/NLBM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
sudo rm -rf build
