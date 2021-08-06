cd /home/recommenders-addons/demo/movielens-100k-estimator
rm nohup.out
nohup sh train.sh &
tail -f nohup.out
