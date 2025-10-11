# tools/metric.py (修复除零问题)
import numpy as np

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        # 初始化混淆矩阵为全零
        self.confusion_matrix = np.zeros((self.num_class,) * 2, dtype=np.int64) # 使用整数类型
        self.eps = 1e-8 # 用于避免除以零的小常数 (主要用于全局指标)

    def _generate_matrix(self, gt_image, pre_image):
        """根据真实标签和预测结果生成单张图像的混淆矩阵"""
        # 确保掩码内的值在有效类别范围内
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        # 计算混淆矩阵的索引
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        # 计算每个组合出现的次数
        count = np.bincount(label, minlength=self.num_class ** 2)
        # 将一维计数转换为二维混淆矩阵
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        """累加批次数据的混淆矩阵"""
        # 确保输入形状一致
        assert gt_image.shape == pre_image.shape, f'pre_image shape {pre_image.shape}, gt_image shape {gt_image.shape}'
        # 累加混淆矩阵
        # 注意：确保传入的 gt_image 和 pre_image 是 NumPy 数组
        self.confusion_matrix += self._generate_matrix(gt_image.astype(np.int64), pre_image.astype(np.int64))

    def reset(self):
        """重置混淆矩阵"""
        self.confusion_matrix = np.zeros((self.num_class,) * 2, dtype=np.int64)

    # --- 修改后的指标计算方法 ---

    def Precision(self):
        """计算每个类别的精确率 (Precision)"""
        # Precision = TP / (TP + FP) = 对角线元素 / 预测为该类的总数 (列和)
        tp = np.diag(self.confusion_matrix)
        tp_plus_fp = self.confusion_matrix.sum(axis=0) # 列和
        # 安全除法：分母为0时结果为0
        precision = np.divide(tp, tp_plus_fp, out=np.zeros_like(tp, dtype=float), where=tp_plus_fp != 0)
        return precision

    def Recall(self):
        """计算每个类别的召回率 (Recall)"""
        # Recall = TP / (TP + FN) = 对角线元素 / 真实为该类的总数 (行和)
        tp = np.diag(self.confusion_matrix)
        tp_plus_fn = self.confusion_matrix.sum(axis=1) # 行和
        # 安全除法：分母为0时结果为0
        recall = np.divide(tp, tp_plus_fn, out=np.zeros_like(tp, dtype=float), where=tp_plus_fn != 0)
        return recall

    def F1(self):
        """计算每个类别的 F1 Score"""
        # 先安全地计算 Precision 和 Recall
        precision = self.Precision()
        recall = self.Recall()
        # F1 = 2 * (Precision * Recall) / (Precision + Recall)
        denominator = precision + recall
        # 安全除法：分母为0时结果为0
        f1 = np.divide(2.0 * precision * recall, denominator, out=np.zeros_like(denominator, dtype=float), where=denominator != 0)
        return f1

    def Intersection_over_Union(self):
        """计算每个类别的交并比 (IoU)"""
        # Intersection = TP = 对角线元素
        tp = np.diag(self.confusion_matrix)
        # Union = TP + FP + FN = 行和 + 列和 - TP
        union = self.confusion_matrix.sum(axis=1) + self.confusion_matrix.sum(axis=0) - tp
        # 安全除法：分母为0时结果为0
        iou = np.divide(tp, union, out=np.zeros_like(tp, dtype=float), where=union != 0)
        return iou

    def Dice(self):
        """计算每个类别的 Dice 系数"""
        # Dice = 2 * TP / ((TP + FP) + (TP + FN)) = 2 * TP / (列和 + 行和)
        tp = np.diag(self.confusion_matrix)
        denominator = self.confusion_matrix.sum(axis=0) + self.confusion_matrix.sum(axis=1)
        # 安全除法：分母为0时结果为0
        dice = np.divide(2.0 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator != 0)
        return dice

    def OA(self):
        """计算总体准确率 (Overall Accuracy)"""
        # OA = Correctly Classified Pixels / Total Pixels
        # OA = sum(diag(Confusion Matrix)) / sum(Confusion Matrix)
        oa = np.diag(self.confusion_matrix).sum() / (self.confusion_matrix.sum() + self.eps)
        return oa

    def Pixel_Accuracy_Class(self):
        """计算每个类别的像素准确率 (似乎是 Precision 的别名?)"""
        # Acc = TP / (TP + FP) = 对角线 / 列和
        # 与 Precision 计算相同
        acc = self.Precision()
        return acc

    def Frequency_Weighted_Intersection_over_Union(self):
        """计算频率加权的 IoU (FWIoU)"""
        # freq = 每个类别的真实像素数 / 总像素数 = 行和 / 总和
        freq = np.sum(self.confusion_matrix, axis=1) / (np.sum(self.confusion_matrix) + self.eps)
        # 计算每个类别的 IoU (已修复)
        iou = self.Intersection_over_Union()
        # FWIoU = sum(freq * IoU)
        # 仅考虑频率大于0的类别进行加权求和
        fw_iou = (freq[freq > 0] * iou[freq > 0]).sum()
        return fw_iou

    # get_tp_fp_tn_fn 方法已移除，因为 TN 计算不正确且未在核心指标中使用

# --- 测试代码 ---
if __name__ == '__main__':
    # 示例用法
    gt = np.array([[0, 2, 1],
                   [1, 2, 1],
                   [1, 0, 1]], dtype=np.int64)

    pre = np.array([[0, 1, 1],
                    [2, 0, 1],
                    [1, 1, 1]], dtype=np.int64)

    evaluator = Evaluator(num_class=3)
    evaluator.add_batch(gt, pre)

    print("Confusion Matrix:\n", evaluator.confusion_matrix)
    # print("TP, FP, TN, FN:\n", evaluator.get_tp_fp_tn_fn()) # 已移除
    print("Precision per class:\n", evaluator.Precision())
    print("Recall per class:\n", evaluator.Recall())
    print("IoU per class:\n", evaluator.Intersection_over_Union())
    print("OA:\n", evaluator.OA())
    print("F1 per class:\n", evaluator.F1())
    print("Dice per class:\n", evaluator.Dice())
    print("Frequency Weighted IoU:\n", evaluator.Frequency_Weighted_Intersection_over_Union())

    # 测试除零情况
    evaluator.reset()
    gt_zero = np.zeros((3,3), dtype=np.int64)
    pre_zero = np.zeros((3,3), dtype=np.int64)
    evaluator.add_batch(gt_zero, pre_zero) # 添加只有背景的样本
    print("\nTesting zero division case (only background):")
    print("Confusion Matrix:\n", evaluator.confusion_matrix)
    print("IoU per class:\n", evaluator.Intersection_over_Union()) # 其他类别应为 0
    print("F1 per class:\n", evaluator.F1()) # 其他类别应为 0
