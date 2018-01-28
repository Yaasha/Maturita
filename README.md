# Maturita

Character recognition CNN with server based interface.

## Dataset

The CNN model was trained on [*EMNIST*](https://www.nist.gov/itl/iad/image-group/emnist-dataset) dataset.
My best result on the dataset after trying several different setups was **88%** accuracy. 
Results on my own handwriting were around **60%**.

## Dependencies

Python 3.5.4, packages from `requirements.txt` and graphviz.

## Usage

To create/load CNN model and train it run: `python nerual_network.py`.  
> **NOTE:** When creating the model you might experience lags, these should only last for few minutes until the training starts.  

To start the interface run: `python manage.py runserver 0.0.0.0:8000`.  
You can then access the interface by visiting http://localhost:8000.  
If you want to try different settings for the CNN or change filepaths, change the values in `config.py`.

