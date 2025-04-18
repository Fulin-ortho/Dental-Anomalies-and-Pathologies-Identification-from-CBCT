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

# 接口文档：`/cbct/seg`

## 接口描述
用于上传 `.nii` 或 `.nii.gz` 格式的文件，进行分割处理，并返回分割后的文件。

## 请求方式
**POST**

## 请求路径
`http://test.zbeiyun.com:59997/cbct/seg`

## 请求参数

| 参数名 | 类型   | 是否必填 | 描述                                          |
| ------ | ------ | -------- | --------------------------------------------- |
| `file` | `File` | 是       | 上传的文件，必须为 `.nii` 或 `.nii.gz` 格式。 |

## 响应参数

| 参数名    | 类型      | 描述                                               |
| --------- | --------- | -------------------------------------------------- |
| `msg`     | `string`  | 返回的消息内容，描述请求的处理结果。               |
| `data`    | `object`  | 返回的数据内容，通常为 `None`。                    |
| `code`    | `int`     | 返回的状态码，标识请求的处理状态：                 |
|           |           | - `2002`：没有文件被上传。                         |
|           |           | - `2003`：未选择文件。                             |
|           |           | - `2004`：文件格式错误（非 `.nii` 或 `.nii.gz`）。 |
|           |           | - `2005`：分割失败。                               |
| `success` | `boolean` | 请求是否成功，`true` 或 `false`。                  |

## 响应示例

### 成功响应
成功时，接口会直接返回分割后的文件供下载，文件名为 `{原始文件名}`。

#### 示例
```
HTTP 响应头：Content-Disposition: attachment; filename="xxx.nii.gz" Content-Type: application/octet-stream
文件内容为分割后的 `.nii` 或 `.nii.gz` 文件。
```

### 错误响应

#### 未上传文件或者参数错误
```json
{
    "msg": "没有文件被上传！",
    "data": null,
    "code": 2002,
    "success": false
}
```

#### 未选择文件
```json
{
    "msg": "未选择文件！",
    "data": null,
    "code": 2003,
    "success": false
}
```

#### 文件格式错误
```json
{
    "msg": "文件格式错误！",
    "data": null,
    "code": 2004,
    "success": false
}
```

#### 分割失败
```json
{
    "msg": "分割失败！",
    "data": null,
    "code": 2005,
    "success": false
}
```
