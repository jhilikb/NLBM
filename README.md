# NLBM
#  Clone the repository and cd into it 
1. git clone https://github.com/jhilikb/NLBM.git
2. cd  NLBM
# run the file docker/launchenv.sh and build it
3. sh docker/launchenv.sh
4. sh /home/NLBM/docker/build.sh
# for testing images you can run the following (you can process 1 or more images at once)
4. cd /home/NLBM/build/bin
5. python test.py 'path_to_input_image_dir' 'path_to_save_output_image_dir' 'model_checkpoint_dir'
6. ex.   python test.py /home/NLBM/net/images/ /home/NLBM/net/output /home/NLBM/net
7. You can download the pretrained model from https://drive.google.com/file/d/1-Epmgn1eIngVNri1ZqbMOz8nhifd3Qp8/view?usp=sharing
# for training the network you can use 
7. cd /home/NLBM/build/bin
8. python blur_train.py 'path_where_input_and_target_is_stored' 'path_to_checkpoint_dir'
9. You can download CUHK input and targets from https://drive.google.com/drive/folders/1quTfs76msCnXjSnDtyr3KTvpQJg6ObnB?usp=sharing

