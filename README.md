Datasets are grouped in the `/home/awad/dataset` directory. This includes the final datasets used for training as well as the intermediate ones obtained through each one of the dataprep and preprocessing phases. Some datasets go through dataprep before preprocessing, this is usually for one hot encoding (ohe) or the removal of side information (in case of Windows Benign dataset). In preprocessing we do oversample, undersample, followed by multiple shuffles.

# Directory listing
```
/home/awad/dataset
|
└─── backups/
└─── dataset_lin_all/						OHE Linux dataset
└─── dataset_lin_shuffled/					Final Linux datasets, used for training
└─── dataset_win_all/						Preprocessing of Windows dataset
└─── dataset_win_benign/					Benign Windows traces
└─── dataset_win_mw/						Malware Windows traces
└─── dataset_win_shuffled/				 	Final Windows datasets, used for training
|    ohe_data.npy
|    ohe_labels.npy
└─── oversample_undersample_and_shuffle/	Preprocessing scripts
└─── prep_lin_all/							Preprocessing of Linux dataset
└─── prep_lin_benign/						Dataprep of Linux Benign dataset
└─── prep_win_mw/							Dataprep of Windows Malware traces

|
```

# Dataprep
## Linux Benign
Linux benign traces are retrieved via Elasticsearch based on <VMID, PID> pair. On the Passau forensic machine we transform these Elasticsearch documents into npy files, one hot encoded. We retrieve those npy files via scp and store them in the `ohe_dataset_benign` directory. First step in dataprep is to use the `prepdataohe.py` script to group them into one big npy array. We also do zero padding, so that the resulting npy array has uniform dimensions. It is necessary to zero pad again when grouping malware and benign data together. 
```
/home/awad/dataset
└─── prep_lin_benign/						Dataprep of Linux Benign dataset
|    | ohe_data_benign.npy					(65, 393, 80)
└─── ohe_dataset_benign/					Benign traces grouped by <VMID, PID>
|    | ohe_labels_benign.npy				(65,)
|    | prepdataohe.py						Group benign samples into one npy array, zero padded
```

## Windows Malware
First step is to run `prepdataohe_mw.py` script to generate `traces_win_mw_ohe.npy`, this does one hot encoding based on the `ntargs.txt` that lists all Windows syscalls. Then we pass this numpy file to `paddata_ohe.py` to do zero padding and crop sequences to a length of 10000. The result is 2 numpy files `ohe_data_win_mw.npy`, `ohe_labels_win_mw.npy` that are copied over to the `~/dataset/dataset_win_mw/` directory.
```
└─── prep_win_mw/							Dataprep of Windows Malware traces
|    | ntargs.txt
|    | paddata_ohe.py						Step 2, zero pad and crop
|    | prepdataohe_mw.py					Step 1, one hot encode
|    | statistics.win.mw					Sample output from the script with the same name
|    | statistics_win_mw.py					Prints the sequence length of samples in the `win_mw_rawdata` directory
|    | traces_win_mw_ohe.npy				(92, ), irrelevant
└─── win_mw_rawdata/						txt files retrieved from zooshare
|
```

# Preprocessing


# Final Datasets

## Windows
Undersampled, dimensions are 44x250x596
Reccommended batch size is 11
```
/local_home/awad/dataset/dataset_win_shuffled/data1.npy
/local_home/awad/dataset/dataset_win_shuffled/labels1.npy

/local_home/awad/dataset/dataset_win_shuffled/data2.npy
/local_home/awad/dataset/dataset_win_shuffled/labels2.npy

/local_home/awad/dataset/dataset_win_shuffled/data3.npy
/local_home/awad/dataset/dataset_win_shuffled/labels3.npy
```
## Linux
Undersampled, dimensions are 90x250x671
Reccommended batch size is 10
```
/local_home/awad/dataset/dataset_lin_shuffled/d_1.npy
/local_home/awad/dataset/dataset_lin_shuffled/l_1.npy

/local_home/awad/dataset/dataset_lin_shuffled/d_2.npy
/local_home/awad/dataset/dataset_lin_shuffled/l_2.npy

/local_home/awad/dataset/dataset_lin_shuffled/d_3.npy
/local_home/awad/dataset/dataset_lin_shuffled/l_3.npy
```
