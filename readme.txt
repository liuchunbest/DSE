This project is to extract software features from public textual product descriptions by integrating overlapping sentence clusters detection and bigram collocation elication. 

Some notes about the programs are as follows:

    1.The data scraped from the website of softPedia and the results of the evaluation are in the folder of data. In detail, the filtered data from softPedia are in the folder of filted_25; the sampled data for evaluation is in the folder of selected_25_100.

    2.The programs about preprocessing the data are in the file of text_process. The programs about detecting overlapping sentence clusters are in the file of detect_cluster. The programs about selecting a given number of clusters are in the file of select_cluster. The programs about extracting bigram collocations from selected clusters are in the file of extract_feature. 

    3.When you run the programs, the path in the file of parameters should be first modified. The path is the complete file path of the project. The programs are writed with python. Some related packages are required. 

    4.When running the program, you should first specify which categrory of products you are considering. For this purpose, you should find the parameter of dataset_id in the file of main and modify it.The value of it means the order of the selected category in the file list of selected_25_100. There are several parameters in the main file. You can also modify them. These parameters are introduced in the paper. 

    5.These programs have been writted by different persons and for a long time. So there are inconsistencies for the style of naming varables and functions. If you have interests in such programs and there are some problems, please write the email to liuchun@henu.edu.cn.

    6.We would be grateful if you find some defects about the programs and send them to us. 
