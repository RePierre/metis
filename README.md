# metis #

## Starting point ##

We started with the following network:

![LSTM network](./img/initial-graph.png "The first LSTM network")

which at the first run did not do so well:

![Loss](./img/initial-loss.png "The initial plotting of our loss function")


### First run ###

The network was trained on `sts-dev.csv` file with 100 epochs and a batch size of 1.
``` shell
python lstm.py --input-file ../../data/sts-dev.csv \
	           --batch-size 1 \
			   --epochs 100
```
