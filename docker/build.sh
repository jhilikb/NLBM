apt update 
apt-get install -y python3-opencv
pip install opencv-python
cd /home/NLBM
mkdir build
cd build
mkdir bin
cd bin
cp /home/NLBM/model/* .
cp /home/NLBM/python/* .
