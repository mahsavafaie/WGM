# WGM
A tool for machine-printed and handwritten text separation at the level of pixels. 

In order to modify or execute individual files, create a new conda environment with the file environment.yml. After activating the newly created environment, you should be able to run and modify existing files without dependency problems.

Two different models can be found in the models/ folder. "scanned" works best with scanned document images and "microfilm" works best with "miccrofilm" images. 

The script classifier_fcnn.py can be used to classify single images. To classify multiple images use 

```for i in /directory/to/multiple/images/*; do python classifier_fcnn.py -i $i -o /directory/to/multiple/images/ --enableCRF; done```



