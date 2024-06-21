The 'images' directory contains plots/figures/results of the analyses made.
The 'docs' directory contains the different versions of the thesis.
The 'bronnen' directory contains some of the literature used.
The 'scripts' directory contains the important scripts used for this thesis. Inside it there is a 'helper_scripts' folder, which is used for evaluation, making plots, analyses etc. These are not important for the process itself.
The important scripts are 'create_table.py' and 'coco_annotations.py'. The first is used to create an Excel table from the rule-based method. The latter is used to create pseudo-annotations from the rule-based method.
The 'model' directory contains the Detectron2 notebook, which is a tutorial for Detectron2 usage, which I edited and used for the layout parser model. In this directory, you should add the config .yml file and the weights for the model you want to use. 
Files I used: 
config: https://drive.google.com/file/d/1qjfrdLPd9OzWdpMYa-8coe1AUdkCaxd5/view?usp=sharing
weights: https://drive.google.com/file/d/1-Pc2P2Rb6wvHnORrH8hUHqypMBGYzlu1/view?usp=sharing
Furthermore, here you should add a dataset subdirectory, with two folders: train and val. Here, you should place the .json COCO annotation file and the images for both the train and validation sets.
