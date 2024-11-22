import numpy as np


class CalUtils():

    @staticmethod
    def jaccard(y_true, y_pred):
        """
        计算Jaccard相似系数,|D∩G|/|D∪G|,该数值越接近1代表越准确
        :param y_true: 真实值
        :param y_pred: 预测值
        :return: jaccard数值
        """
        y_true_f = y_true.flatten()  # 展开成一维数组
        y_pred_f = y_pred.flatten()
        intersection = np.sum(y_true_f * y_pred_f)  # 求交集
        union = np.sum(np.logical_or(y_true_f, y_pred_f))  # 求并集
        return intersection / union

    @staticmethod
    def dice_coef(y_true, y_pred):
        """
        计算dice系数（DSC），2 * |D∩G| / |D| + |G|,该数值越接近1代表越准确
        :param y_true:真实值
        :param y_pred: 预测值
        :return: 2倍的真实值和预测值的交集/真实值+预测值
        """
        y_true_f = y_true.flatten()  # 展开成一维数组
        y_pred_f = y_pred.flatten()
        intersection = np.sum(y_true_f * y_pred_f)
        return (2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f))

    @staticmethod
    def rvd(y_true, y_pred):
        """
        计算相对体积差（RVD），|D - G|/|G|, 该数值越接近0代表越准确
        :param y_true: 真实值
        :param y_pred: 预测值
        :return: rvd系数
        """
        y_true_f = y_true.flatten()  # 展开成一维数组
        y_pred_f = y_pred.flatten()
        difference = np.sum(np.abs(y_pred_f - y_true_f))
        return difference / np.sum(y_true_f)

    @staticmethod
    def fa(y_true, y_pred):
        """
        检测准确度（FA）, |D| / |D∪G|
        :param y_true: 真实值
        :param y_pred: 预测值
        :return: FA数据
        """
        y_true_f = y_true.flatten()  # 展开成一维数组
        y_pred_f = y_pred.flatten()
        union = np.sum(np.logical_or(y_true_f, y_pred_f))  # 求并集
        return np.sum(y_pred_f) / union
