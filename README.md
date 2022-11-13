# hugadb_cnn

# download data at this [link](https://drive.google.com/drive/folders/12LG7_hsxEBULu-hKwP5NTgnsTcGON8JV?usp=share_link)
### Running steps with the HugoDB dataset
Step 1: Download the dataset from the link above. Copy the dataset to the Data folder. Create a TransformData folder to transform data.
Step 2: Run extract_data.py file to transform data. Run the file with the following command:

`python extract_data.py`

Step 3: Run the file concat_file.py to merge data from the files. Run the file with the following command:

`python concat_file.py`

Step 4: Run baseline test with svm algorithm.

`python svm_raw.py`

Step 5: Run CNN model has been optimized.

`python cnn_pca.py`
