Datasets are grouped in the `/home/awad/dataset` directory. This includes the final datasets used for training as well as the intermediate ones obtained through each one of the dataprep and preprocessing phases. Some datasets go through dataprep before preprocessing, this is usually for one hot encoding (ohe) or the removal of side information (in case of Windows Benign dataset). In preprocessing we do oversample, undersample, followed by multiple shuffles.

# Directory listing
```
/home/awad/dataset
|
└─── backups/
└─── dataset_lin_all/	                  OHE Linux dataset
└─── dataset_lin_shuffled/                Final Linux datasets, used for training
└─── dataset_win_all/                     Preprocessing of Windows dataset
└─── dataset_win_benign/                  Benign Windows traces
└─── dataset_win_mw/                      Malware Windows traces
└─── dataset_win_shuffled/                Final Windows datasets, used for training
|    ohe_data.npy
|    ohe_labels.npy
└─── oversample_undersample_and_shuffle/  Preprocessing scripts
└─── prep_lin_all/                        Preprocessing of Linux dataset
└─── prep_lin_benign/                     Dataprep of Linux Benign dataset
└─── prep_win_mw/                         Dataprep of Windows Malware traces
|
```

# Dataprep
## Linux Benign
Linux benign traces are retrieved via Elasticsearch based on <VMID, PID> pair. On the Passau forensic machine we transform these Elasticsearch documents into npy files, one hot encoded. We retrieve those npy files via scp and store them in the `ohe_dataset_benign` directory. First step in dataprep is to use the `prepdataohe.py` script to group them into one big npy array. We also do zero padding, so that the resulting npy array has uniform dimensions. It is necessary to zero pad again when grouping malware and benign data together. 
```
/home/awad/dataset
└─── prep_lin_benign/           Dataprep of Linux Benign dataset
|    | ohe_data_benign.npy      (65, 393, 80)
└─── ohe_dataset_benign/        Benign traces grouped by <VMID, PID>
|    | ohe_labels_benign.npy    (65,)
|    | prepdataohe.py           Group benign samples into one npy array, zero padded
|
```

## Windows Malware
First step is to run `prepdataohe_mw.py` script to generate `traces_win_mw_ohe.npy`, this does one hot encoding based on the `ntargs.txt` that lists all Windows syscalls. Then we pass this numpy file to `paddata_ohe.py` to do zero padding and crop sequences to a length of 10000. The result is 2 numpy files `ohe_data_win_mw.npy`, `ohe_labels_win_mw.npy` that are copied over to the `~/dataset/dataset_win_mw/` directory.
```
└─── prep_win_mw/                 Dataprep of Windows Malware traces
|    | ntargs.txt
|    | paddata_ohe.py             Step 2, zero pad and crop
|    | prepdataohe_mw.py          Step 1, one hot encode
|    | statistics.win.mw          Sample output from the script with the same name
|    | statistics_win_mw.py       Prints the sequence length of samples in the `win_mw_rawdata` directory
|    | traces_win_mw_ohe.npy      (92, ), irrelevant
└─── win_mw_rawdata/              txt files retrieved from zooshare
|
```

# Preprocessing
In preprocessing, we do three main things
1.  Group benign and malware samples into one <dataset, labels> pair
2.  Oversample or Undersample 
3.  Shuffle

## Scripts
Scripts to perform these actions are in the following directory.
```
oversample_undersample_and_shuffle/
|    | oversample.py
|    | shuffle.py
|    | undersample.py
|
```

## Windows dataset
The result of dataprep is a numpy pair for malware data and another one for benign data. This is true for both Windows and Linux. First step here is to regroup those using the `datasetwinregroup.py` script. Then the result undergoes oversampling and undersampling, both having their own directories. After that, we shuffle, the result is stored in `oversampling/shuffles` and `undersampling/shuffles`. Each shuffle has been trained, at the end 3 shuffles were selected and copied over to the `~/dataset/dataset_win_shuffled` to be used for the experiments 1 and 2.

A special case in Windows dataset is to remove side information, this is why we use the `subseq.py` script to trim the third dimension and remove side information from it. However, no need to reuse it the Windows benign data is save at `~/dataset/dataset_win_benign/win_benign_traces_noparams.npy` and `~/dataset/dataset_win_benign/win_benign_labels_noparams.npy`.

```
└─── ~/dataset/dataset_win_all/
|    | data2.npy
|    | data3.npy
|    | data.npy
|    | data_ros.npy
|    | datasetwinregroup.py           Regroup windows dataset malware and benign
|    | labels3.npy
|    | labels.npy
|    | labels_ros.npy
└───  oversampling/
|    | data_oversampled_ros.npy       Oversampled data using Random Oversampling (ros)
|    | labels_oversampled_ros.npy
|    | shuffle.py                     shuffle script copied over from `oversample_undersample_and_shuffle`
|    └───  shuffles/                  Directory containing multiple shuffles, all can be used for training
|    | subseq.py                      Script to cut elements from the third dimension.
|
└───  undersampling/
|    | data_undersampled_nm.npy       Undersampled data using Near Miss
|    | data_undersampled_rus.npy      Undersampled data using Random Undersampling (rus)
|    | labels_undersampled_nm.npy
|    | labels_undersampled_rus.npy
|    └─── shuffles/                   Directory containing multiple shuffles, all can be used for training
|
```

## Linux dataset
The result of oversampling and undersampling is stored in the following directory.
```
└─── ~/dataset/dataset_lin_shuffled/
|    └─── allargs/              (90, 250, 2188)
|    | d_1.npy                  (90, 250, 671)
|    | d_2.npy
|    | d_3.npy
|    | d_3.npy.1
|    | d_3_oversampled.npy
|    | l_1.npy
|    | l_2.npy
|    | l_3.npy
|    | l_3_oversampled.npy
|
```
To be able to run this, we exclude the first argument. As a backup, d_1, d_2, and d_3 containing arg1 have been stored in `allargs`. Notice the dimensions above.

The oversampling and undersampling are performed and stored in `~/dataset/prep_lin_all`. To create more benign samples, more benign tracing was conducted. Before that, the Linux dataset contained 73 malware samples and 2 benign samples, these are stored as a backup in `dataset_73_2`. Same as with Windows, the oversampled and undersampled results are stored in their respective directories, which then undergoes shuffling. 

```
└─── ~/dataset/prep_lin_all/
|    | data_regroup.py
|    └─── dataset_73_2/
|    | lin_cropped.npy          (171, 250, 2188) Oversampled, cropped to 250m arg1 included
|    | lin_labels_cropped.npy
|    | ohe_data_lin.npy         (171, 2463, 2188) Oversampled, arg1 included
|    | ohe_data.npy             (75, 83, 1011) Dataset with 73 malware and 2 benign samples
|    └─── ohe_dataset/
|    | ohe_labels_lin.npy
|    | ohe_labels.npy
|    └─── outdated/
|    └─── oversample/
|    | prepdataohe.py
|    | sd.py
|    | trace_dim.txt
|    └─── undersample/
```

### Exclusion of arg1
No script is provided for this. To exclude arg1, remove columns with indices `46` to `1563`.


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
