echo "change directory to that for CS231 CNN for computer vision"
cd d:
cd "Wei Ann Lim\Documents\github\Stanford_CS231n_CNN\assignment2\floydhub"
floyd login
y
eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJodHRwczovL2Zsb3lkaHViLmF1dGgwLmNvbS8iLCJzdWIiOiJhdXRoMHw1OWFmNTFlMWQyNWM4ZjVmN2M5OTJjZDQiLCJhdWQiOiJRZEl5NlJHM1Z5cVdyMWFmTnNhWVRGRTFDZlVXZnRVTSIsImlhdCI6MTUwOTI4MDU1MywiZXhwIjoxNTA5ODg1MzUzfQ.PWSFMCod3X1LbmkAPGd5Y3qHXGAwuHBaAmXZbr8WhCU
floyd init cs231n
floyd run --data wanormal/datasets/cs231n_cifar10/2:cs231n --gpu --env pytorch-0.2 --mode jupyter