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

Owing to data privacy considerations and ethical constraints, the datasets and trained model weights cannot be made publicly available. To enable more extensive scientific research and practical applications, we offer an API - based inference solution.

To utilize this solution, users can employ Postman to access the URL http://test.zbeiyun.com:59997/cbct/seg. Here, they need to upload a CBCT zip 'file'. Once uploaded, the resulting outputs can be easily processed and downloaded. Subsequently, these outputs can be further utilized with medical imaging software like ITK - SNAP (accessible at https://www.itksnap.org/pmwiki/pmwiki.php).

In addition, a preliminary commercial version of the segmentation service is accessible online at https://www.aortho360.com. For validation or testing requirements, the authors are willing to provide temporary user accounts. Interested parties can request these accounts by contacting Fulin Jiang at jfl@cqu.edu.cn.
