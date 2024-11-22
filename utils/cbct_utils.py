import SimpleITK as sitk
import os
import numpy as np
import cv2
from scipy import ndimage
import vtk
import sympy as sp

base_dir = os.path.dirname(os.path.dirname(__file__))


def dcm2nii(path_read, path_save):
    """
    将批量的dcm文件转为nii格式
    :param path_read: dcm文件路径
    :param path_save:转换成nii的保存路径
    :return:无返回值
    """
    # GetGDCMSeriesIDs读取序列号相同的dcm文件
    series_id = sitk.ImageSeriesReader.GetGDCMSeriesIDs(path_read)
    # GetGDCMSeriesFileNames读取序列号相同dcm文件的路径，series[0]代表第一个序列号对应的文件
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
        path_read, series_id[0])

    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    image3d = series_reader.Execute()
    sitk.WriteImage(image3d, path_save)


def nii2stl(path: str, nii_data, start_label=1, end_label=48, smoothing_iterations=16, pass_band=0.001, feature_angle=120.0):
    """
    nii文件转换成stl文件(整副牙齿，逐一保存)
    :params path:nii文件路径 
    :params nii_data:nii数据
    :params start_label:nii文件中开始标签 
    :params end_label:nii文件中结束标签 
    :params smoothing_iterations:平滑迭代次数，值越大stl越平滑
    :params pass_band:
    :params feature_angle:特征角度
    :return:
    """
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(path)
    reader.Update()
    image = reader.GetOutput()
    histogram = vtk.vtkImageAccumulate()
    discrete_cubes = vtk.vtkDiscreteFlyingEdges3D()
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    selector = vtk.vtkThreshold()
    scalars_off = vtk.vtkMaskFields()
    geometry = vtk.vtkGeometryFilter()

    # Generate models from labels
    # 1) Read the meta file
    #    -- 也可以是读取imagedata
    # 2) Generate a histogram of the labels
    # 3) Generate models from the labeled volume
    # 4) Smooth the models
    # 5) Output each model into a separate file

    histogram.SetInputData(image)
    histogram.SetComponentExtent(0, end_label, 0, 0, 0, 0)
    histogram.Update()

    # discrete_cubes.SetInputConnection(reader.GetOutputPort())
    discrete_cubes.SetInputData(image)
    discrete_cubes.GenerateValues(
        end_label - start_label + 1, start_label, end_label)

    smoother.SetInputConnection(discrete_cubes.GetOutputPort())
    smoother.SetNumberOfIterations(smoothing_iterations)
    smoother.BoundarySmoothingOn()
    smoother.FeatureEdgeSmoothingOn()
    smoother.SetFeatureAngle(feature_angle)
    smoother.SetPassBand(pass_band)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()

    selector.SetInputConnection(smoother.GetOutputPort())
    selector.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject().FIELD_ASSOCIATION_POINTS,
                                    vtk.vtkDataSetAttributes().SCALARS)

    # Strip the scalars from the output
    scalars_off.SetInputConnection(selector.GetOutputPort())
    scalars_off.CopyAttributeOff(vtk.vtkMaskFields().POINT_DATA,
                                 vtk.vtkDataSetAttributes().SCALARS)
    scalars_off.CopyAttributeOff(vtk.vtkMaskFields().CELL_DATA,
                                 vtk.vtkDataSetAttributes().SCALARS)

    geometry.SetInputConnection(scalars_off.GetOutputPort())

    # writer = vtk.vtkXMLPolyDataWriter()
    writer = vtk.vtkSTLWriter()
    writer.SetFileTypeToBinary()  # writer.SetFileTypeToASCII()
    writer.SetInputConnection(geometry.GetOutputPort())

    # start Transformation and save

    try:
        for data in nii_data:
            if data['defect']:
                continue
            else:
                # select the cells for a given label

                selector.ThresholdBetween(data['start'], data['end'])

                # output the polydata

                writer.SetFileName(data['stl_path'])
                writer.Write()
    except Exception as e:
        print(e)




def raw2nii(path: str, save_path: str):
    """
    raw文件转换为nii格式文件
    :param save_path: 转换之后的文件保存路径
    :param path:raw文件路径
    :return: 无返回值
    """
    img_data = np.fromfile(path, dtype='uint16')
    _, filename = os.path.split(path)
    shortname, _ = os.path.splitext(filename)
    name_split = filename.split('_')
    print(img_data.shape)
    data_new_shape = img_data.reshape(
        int(name_split[3]), int(name_split[2]), int(name_split[1]))
    out = sitk.GetImageFromArray(data_new_shape)
    sitk.WriteImage(out, os.path.join(save_path, shortname + '.nii.gz'))


def nii2img(path):
    """
    cbct文件到处为图像文件
    :param path: nii文件路径
    :return: 无返回值
    """
    ct = sitk.ReadImage(path)
    _, file = os.path.split(path)
    filename = str(file).split(".")[0]
    savepath_x = os.path.join(r'F:\DATASET\cbct_u_2_dataset_x\label', filename)
    savepath_y = os.path.join(r'F:\DATASET\cbct_u_2_dataset_y\label', filename)
    savepath_z = os.path.join(r'F:\DATASET\cbct_u_2_dataset_z\label', filename)
    # 截取掉最后的uint之后的字符
    # image
    # savepath_x = savepath_x[:-7]
    # savepath_y = savepath_y[:-7]
    # savepath_z = savepath_z[:-7]
    # label
    savepath_x = savepath_x[:-10]
    savepath_y = savepath_y[:-10]
    savepath_z = savepath_z[:-10]

    if not os.path.exists(savepath_x):
        os.makedirs(savepath_x)
    if not os.path.exists(savepath_y):
        os.makedirs(savepath_y)
    if not os.path.exists(savepath_z):
        os.makedirs(savepath_z)
    # 读取图像
    ct_array = sitk.GetArrayFromImage(ct)

    ct_array[ct_array < 0] = 0
    ma = np.max(ct_array)
    ct_array = ct_array / ma * 255.

    # ct_array[ct_array > 0] = 255
    z, y, x = ct_array.shape

    for i in range(z):
        cv2.imwrite(os.path.join(savepath_z, str(i) + '.png'),
                    ct_array[i, :, :])

    for i in range(y):
        cv2.imwrite(os.path.join(savepath_y, str(i) + '.png'),
                    ct_array[:, i, :])

    for i in range(x):
        cv2.imwrite(os.path.join(savepath_x, str(i) + '.png'),
                    np.flipud(ct_array[:, :, i]))


def sitk_read_raw(img_path, resize_scale=1):
    """
    读取3D图像并resale（因为一般医学图像并不是标准的[1,1,1]scale）
    :param img_path: cbct图像路径
    :param resize_scale: sacle尺度，默认为1
    :return:scale之后的numpy数据
    """
    nda = sitk.ReadImage(img_path)
    if nda is None:
        raise TypeError("input img is None!!!")
    nda = sitk.GetArrayFromImage(nda)  # channel first

    nda = ndimage.zoom(nda, [resize_scale, resize_scale,
                             resize_scale], order=0)  # rescale

    return nda


def resize_image_itk(itkimage, newSize, np_array=False, resamplemethod=sitk.sitkNearestNeighbor):
    """
    将医学图像不改变整体形状的情况下resize成指定形状 形状为(x,y,z)
    :param itkimage: itk图像数据
    :param newSize: 需要resize的尺寸
    :param np_array: 是否返回numpy数组
    :param resamplemethod: 差值方式，默认使用临近差值（mask使用临近差值，imag使用线性插值）
    :return:处理之后的图像数据，numpy数组或itk数据
    """
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()  # 原来的体素块尺寸
    originSpacing = itkimage.GetSpacing()
    newSize = np.array(newSize, float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int16)  # spacing肯定不能是整数
    resampler.SetReferenceImage(itkimage)  # 需要重新采样的目标图像
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)  # 得到重新采样后的图像
    if np_array:
        return sitk.GetArrayFromImage(itkimgResampled)
    else:
        return itkimgResampled



def dist_pos(c1, c2):
    """
    判断坐标在第几象限（牙齿上的象限）
    :params c1:需要判断的坐标 （x,y,z）
    :params c2:坐标原点 （x,y,z）
    :return:第几象限
    """
    x, _, z = c1
    o_x, _, o_z = c2
    if z >= o_z and x <= o_x:
        return 1
    elif z >= o_z and x >= o_x:
        return 2
    elif z <= o_z and x <= o_x:
        return 4
    else:
        return 3


def trans_2d(img_3d):
    """
    先将图像读取出来生成2D数据
    :param img_3d: cbct的numpy数据
    :return: 返回横断面,冠状面的numpy图像数据,3d图像的形状
    """
    # 从z方向上获取图像
    z_img = np.max(img_3d, 0)
    z_img = z_img / np.max(np.abs(z_img)) * 255
    z_img[z_img < 0] = 0
    z_img = z_img.astype(np.uint8)
    # 从z方向上获取图像
    y_img = np.max(img_3d, 1)
    y_img = y_img / np.max(np.abs(y_img)) * 255
    y_img[y_img < 0] = 0
    y_img = y_img.astype(np.uint8)

    return z_img, y_img, img_3d.shape

def draw_umich_gaussian(heatmap, center, radius, k=1):
    """
    根据中心点绘制每一个类别的高斯热图
    :params heatmap:需要绘制的热图 
    :params center:需要进行高斯模糊的中心点 
    :return:生成的热图
    """
    diameter = 2 * radius + 1  # 高斯核直径
    gaussian = gaussian_3d((diameter, diameter, diameter), sigma=diameter / 6)

    x, y, z = int(center[0]), int(center[1]), int(center[2])
    w, h, d = heatmap.shape  # 三个状面

    # 计算中心点距离边缘的情况，以便于确认高斯模式的范围
    left, right = min(x, radius), min(w-x, radius+1)
    fontend, backend = min(y, radius), min(h-y, radius+1)
    bottom, top = min(z, radius), min(d-z, radius+1)

    masked_heatmap = heatmap[x-left:x+right,
                             y-fontend:y+backend, z-bottom:z+top]

    masked_gaussian = gaussian[radius-left:radius+right,
                               radius-fontend:radius+backend, radius-bottom:radius+top]

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        # 逐元比较两元素大小并且替换masked_heatmap中的值
        np.maximum(masked_heatmap, masked_gaussian*k, out=masked_heatmap)
    return heatmap

def gaussian_radius(det_size, min_overlap=0.7):
    """
    生成高斯模糊半径
    :params det_size:tuple of int (w,h,d)
    :params min_overlap:两个目标的最小重叠区
    :return:float 类型的高斯半径
    """

    w, h, d = det_size
    # 求出一元三次方程的常数项
    a1 = 1
    b1 = -(w+h-d)
    c1 = (h*w)+(h*d)+(w*d)
    d1 = w*h*d*(min_overlap-1)/(1+min_overlap)

    r1 = solve_cubic_equation(a1, b1, c1, d1)

    a2 = 8
    b2 = -4*(w+h+d)
    c2 = 2*(w*h+w*d+h*d)
    d2 = w*h*d*(min_overlap-1)
    r2 = solve_cubic_equation(a2, b2, c2, d2)

    a3 = 8*min_overlap
    b3 = 4*min_overlap*(w+h+d)
    c3 = 2*min_overlap*(h*w+h*d+w*d)
    d3 = w*h*d*(min_overlap-1)
    r3 = solve_cubic_equation(a3, b3, c3, d3)

    r = min(r1, r2, r3)
    return max(0, r)


def solve_cubic_equation(a, b, c, d):
    """
    求一元三次方程的实根：ax^3+bx^2+cx+d=0
    :params a,b,c,d:为方程中的常数项 
    :return:
    """
    x = sp.Symbol('x')
    f = a*x**3 + b*x**2 + c*x + d
    x = sp.solve(f)
    res = []
    for v in x:
        if isinstance(v, sp.Number):
            res.append(float(v))
    if len(res) == 0:  # 无实数根
        return 0

    return max(res)

def gaussian_3d(shape, sigma=1):
    """Generate 3d gaussian.

    Parameters
    ----------
    shape : tuple of int
        The shape of the gaussian.
    sigma : float
        Sigma for gaussian.

    Returns
    -------
    float
        3D gaussian kernel.

    """

    m, n, l = [(ss - 1.) / 2. for ss in shape]

    y, x, z = np.ogrid[-m:m+1, -n:n+1, -l:l+1]

    h = np.exp(-(x * x + y * y+z * z) / (2 * sigma * sigma))

    h[h < np.finfo(h.dtype).eps * h.max()] = 0

    return h