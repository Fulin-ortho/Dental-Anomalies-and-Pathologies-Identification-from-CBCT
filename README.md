# Deep learning-based dental anomalies and pathologies identification for orthodontic treatment
by Juan Li et al.

## Introduction
This repository is for our paper 'Dental anomalies and pathologies identification for orthodontic treatment: an artificial intelligence-based study'.

## Release
The training, data preparation and testing codes have been released. 

## Inference

Write the full path of the CBCT data (.nii.gz) in the file.list, and set up in the run_test.py.
Run the model:  run_test.py.

## AI inference application

Due to data privacy considerations and ethical constraints, we are unable to publicly release the original datasets or the trained model weights. To facilitate further scientific research and clinical application, we have developed an API-based inference solution. 

Users can upload a CBCT DICOM archive (in zip format) using tools such as Postman to the endpoint at http://test.zbeiyun.com:59997/cbct/seg. Once the file is processed, the segmentation outputs can be easily downloaded and subsequently analyzed using dedicated medical imaging software such as ITK-SNAP (https://www.itksnap.org/pmwiki/pmwiki.php).

Furthermore, a preliminary commercial version of our segmentation service is available at https://www.aortho360.com. For purposes of validation or testing, we are prepared to provide temporary user accounts upon request. Interested parties are invited to contact Fulin Jiang at jfl@cqu.edu.cn for further access and instructions.
