echo "change directory to that for CS231 CNN for computer vision"
cd d:
cd "Wei Ann Lim\Documents\github\Stanford_CS231n_CNN\assignment2\floydhub"
floyd login
floyd init cs231n
floyd run --data wanormal/datasets/cs231n_cifar10/2:cs231n --gpu --env pytorch-0.2 --mode jupyter