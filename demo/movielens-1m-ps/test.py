import tensorflow_datasets as tfds


split_size = int(100 / 2)
split_start = split_size * 1
split = 'train[{}%:{}%]'.format(split_start, split_start + split_size - 1)
print("dataset split, worker{}: {}".format(1, split))
tfds.load("movie_lens/100k-ratings", split=split)