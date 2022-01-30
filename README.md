# ASL-ML
ML model responsible for detecting ASL letters. 

File Descriptions:
  - ImageCreator -> Used the .h5 file from the notebook at [1]. Using tensorflow to load this model and predict realtime hand signs using a webcam. As expected, it is highly inaccurate due to the trainning data not being diverse enough.
  - Dataset_Creation -> Creates a hand sign dataset for image classification. Press on your keyboard the letter you wish to save.
  - Dual_Dataset_Creation -> Has Dataset_creations functionality, but it also creates the Object detection dataset for the hand signs. This dataset is made in COCO format.
  
  
  
References:

[1] https://www.kaggle.com/ryuodan/asl-detection-walkthrough
