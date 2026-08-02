import cv2
import numpy as np
import pandas as pd
import datatable as dt
import matplotlib
import matplotlib.pyplot as plt

def segment_and_annotate(image_path, m=10, n=10, output_csv='grid_labels.csv'):
    img = cv2.imread(image_path)
    if img is None:
        print("错误：无法读取图像")
        return

    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 定义颜色范围
    # 紫色（区域1 - 绿线内部）
    purple_lower = np.array([130, 50, 50])
    purple_upper = np.array([170, 255, 255])
    purple_mask = cv2.inRange(hsv, purple_lower, purple_upper)

    # 蓝色（区域2）
    blue_lower = np.array([100, 50, 50])
    blue_upper = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

    # 红色（区域3）需要合并两个范围
    red_lower1 = np.array([0, 50, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 50, 50])
    red_upper2 = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv, red_lower1, red_upper1) + cv2.inRange(hsv, red_lower2, red_upper2)

    # ========== 关键修复：闭运算 (CLOSE) ==========
    kernel = np.ones((7, 7), np.uint8)  # 加大核尺寸确保闭合缺口
    red_mask_closed = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    blue_mask_closed = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 初始化标签图 (默认全0，即背景)
    label_map = np.zeros((h, w), dtype=int)

    # --- 区域1 (绿线内部 - 紫色填充) ---
    contours, _ = cv2.findContours(purple_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_purple = max(contours, key=cv2.contourArea)
        cv2.drawContours(label_map, [largest_purple], -1, 1, -1)

    # --- 区域2 (蓝线闭合区) ---
    # 在未被区域1占用的区域中找蓝色轮廓
    mask_blue_valid = cv2.bitwise_and(blue_mask_closed, blue_mask_closed, mask=(label_map == 0).astype(np.uint8))
    contours, _ = cv2.findContours(mask_blue_valid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 200:  # 过滤极小点
            cv2.drawContours(label_map, [cnt], -1, 2, -1)

    # --- 区域3 (红线闭合区) ===== 重点修正 =====
    # 在未被区域1和2占用的区域中找红色轮廓
    mask_red_valid = cv2.bitwise_and(red_mask_closed, red_mask_closed, mask=(label_map == 0).astype(np.uint8))
    contours, _ = cv2.findContours(mask_red_valid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 200:
            # 使用闭合轮廓进行填充（-1）
            cv2.drawContours(label_map, [cnt], -1, 3, -1)

    # ====== 新增备选填充方案：如果上述填充失败（如轮廓极细），用洪水填充 ======
    # 找出红色区域内部的一个点作为种子点
    contours_red_fallback, _ = cv2.findContours(red_mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_red_fallback:
        if cv2.contourArea(cnt) > 200:
            # 计算重心
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                # 检查该点是否当前还是 0 (未填充)
                if label_map[cy, cx] == 0:
                    # 使用洪水填充从中心点扩散填充为3
                    mask_flood = np.zeros((h + 2, w + 2), np.uint8)
                    cv2.floodFill(label_map, mask_flood, (cx, cy), 3)

    # ==================== 网格划分 ====================
    data = []
    cell_height = h / m
    cell_width = w / n

    for row in range(m):
        for col in range(n):
            center_y = int(row * cell_height + cell_height / 2)
            center_x = int(col * cell_width + cell_width / 2)
            label = int(label_map[center_y, center_x])

            data.append({
                "row": row,
                "col": col,
                "center_x": center_x,
                "center_y": center_y,
                "label": label
            })
    # ==================== 二维标注保存数据为csv ====================
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False, header=False)
    print(f"处理完成！文件已保存为: {output_csv}")
    print("前10行数据预览：")
    print(df.head(10))
    # ==================== 生成标注图片 ====================
    labels = df.values
    cmap = matplotlib.colors.ListedColormap(['#CCCCCC', '#888888', '#444444', '#000000'])
    plt.figure(figsize=(8, 5), dpi=100)
    plt.axis('off')  # 去坐标轴
    plt.xticks([])  # 去刻度
    plt.yticks([])  # 去刻度
    a = plt.scatter(labels[:, 1],
                    labels[:, 0],
                    c=labels[:, 4],
                    s=1,  # 散点大小
                    cmap=cmap  # 散点颜色集
                    )
    plt.title('')
    plt.show()

def MSI2DlabelCombineCRCto_coords_and_1Dlabels(manuallabel_path, clusterlabel_path, coords_path, labels_path):

    m = np.loadtxt(manuallabel_path, delimiter=',', dtype=int)
    c = np.loadtxt(clusterlabel_path, delimiter=',', dtype=int)
    # 使用manual标注，没有数据的坐标不标注，有数据但没有manual标的用-1标注
    res = np.where((m != 0) & (c != 0), m, np.where((m == 0) & (c != 0), -1, 0))
    np.savetxt('MSI/data/MSIbiaozhu_CRCglycan_combined.csv', res, delimiter=',', fmt='%d')

    # 输入：标注矩阵就是网格
    grid = res
    # 输出文件
    coords_out = coords_path
    labels_out = labels_path

    # 找到所有有效点（标签 != -1）
    rows, cols = np.where(grid != 0)
    # 还原坐标：这里认为网格行列索引即坐标（左上角为 (0,0)）
    # 若已知原始 x_min, y_min，可加上偏移，例如: x = cols + x_min
    x = cols + 100
    y = rows + 12
    labels = grid[rows, cols]

    # 按 x 升序，若 x 相同则按 y 降序排列
    # np.lexsort 中最后的序列为主排序键
    order = np.lexsort((-y, x))  # -y 实现降序
    x_sorted = x[order]
    y_sorted = y[order]
    labels = labels[order]
    labels_sorted = np.where(labels == -1, 0, labels)

    # 写入文件（无表头，逗号分隔）
    # 坐标文件：两列，x, y
    np.savetxt(coords_out, np.column_stack((x_sorted, y_sorted)),
               delimiter=",", fmt="%d")
    # 标签文件：单列
    np.savetxt(labels_out, labels_sorted, delimiter=",", fmt="%d")

    cmap = matplotlib.colors.ListedColormap(['#CCCCCC', '#888888', '#444444', '#000000'])
    plt.figure(figsize=(8, 5), dpi=100)
    plt.axis('off')  # 去坐标轴
    plt.xticks([])  # 去刻度
    plt.yticks([])  # 去刻度
    a = plt.scatter(x_sorted,
                    y_sorted,
                    c=labels_sorted,
                    s=1,  # 散点大小
                    cmap=cmap  # 散点颜色集
                    )
    plt.title('')
    plt.show()


if __name__ == "__main__":
    # ---------------- 运行 -----------------
    # 请将 'your_image.jpg' 替换为你的图片实际路径
    # m=10, n=10 表示将图片分成 10x10 的网格,跟MSI数据像素坐标统一
    segment_and_annotate('MSI/picy.jpg', m=182, n=291, output_csv='MSI/manual_labels.csv')

    MSI2DlabelCombineCRCto_coords_and_1Dlabels(
        'MSI/data/MSIbiaozhu_CRCglycan_manual.csv',
        'MSI/data/MSIbiaozhu_CRCglycan_cluster.csv',
        "MSI/data/coordinateXY_CRCe2.csv",
        "MSI/data/MSIbiaozhuNPArray_CRCglycan_manual_new.csv"
        )
