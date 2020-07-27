import cv2
import numpy as np
import math
import xlwt
import os
import shutil

def cal_distance(pointA_x, pointA_y, pointB_x, pointB_y):
    return math.sqrt((pointA_x-pointB_x)**2 + (pointA_y-pointB_y)**2)


class image():
    """  图片类  """
    def __init__(self, imagename, pixelsPerCM, referenceLength=20):

        self.image = cv2.imread(imagename)
        self.referenceLength = referenceLength
        self.pixelsPerCM = pixelsPerCM

    def image_segmentation(self):
        # 将图片按方格分割成四个部分并计算
        # gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # edged = cv2.Canny(gray, 50, 100)
        # edged = cv2.dilate(edged, None, iterations=1)
        # edged = cv2.erode(edged, None, iterations=1)
        #
        # outimage, contours, hireachy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # for i, cnt in enumerate(contours):
        #     # print(cv2.contourArea(cnt))
        #     if cv2.contourArea(cnt) < 100000:
        #         continue
        #     x, y, w, h = cv2.boundingRect(cnt)
        #     # print(x, y, w, h)
        # self.pixelsPerCM = int(w/self.referenceLength)   # 计算1cm多少像素
        # offset = 40
        # part1_p1_x = int(x + offset); part1_p1_y = int(y + offset)
        # part1_p2_x = int(x + w/2 -offset); part1_p2_y = int(y + h/2 - offset)
        #
        # part2_p1_x = int(x + w/2 + offset); part2_p1_y = int(y + offset)
        # part2_p2_x = int(x + w -offset); part2_p2_y = int(y + h/2 - offset)
        #
        # part3_p1_x = int(x + offset); part3_p1_y = int(y + h/2 +offset)
        # part3_p2_x = int(x + w/2 -offset); part3_p2_y = int(y + h - offset)
        #
        # part4_p1_x = int(x + w/2 + offset); part4_p1_y = int(y + h/2 +offset)
        # part4_p2_x = int(x + w -offset); part4_p2_y = int(y + h - offset)
        #
        # part1 = self.image[part1_p1_y: part1_p2_y, part1_p1_x: part1_p2_x]
        # part2 = self.image[part2_p1_y: part2_p2_y, part2_p1_x: part2_p2_x]
        # part3 = self.image[part3_p1_y: part3_p2_y, part3_p1_x: part3_p2_x]
        # part4 = self.image[part4_p1_y: part4_p2_y, part4_p1_x: part4_p2_x]

        # 直接分割
        part1 = self.image[180:950, 550:1320]
        part2 = self.image[180:950, 1350:2120]
        part3 = self.image[980:1750, 550:1320]
        part4 = self.image[980:1750, 1350:2120]

        return part1, part2, part3, part4


class ProcessMethod():

    """   处理方法类,掉用方法直接返回需要求的值"""
    def __init__(self, src):
        self.src = src              # 原图，画图测试用
        self.width = []             # 香菇茎宽度
        self.MidAxisLength = 0.0    # 香菇茎中轴线长度
        self.diameter = []          # 香菇盖直径
        self.thickness = 0.0        # 香菇盖厚度

    def cal_MidAxis_Width(self):
        # 计算中轴线和宽度的方法

        # 预处理
        convexflag = True
        lower_hsv = np.array([35, 43, 46])
        upper_hsv = np.array([77, 255, 255])
        hsv = cv2.cvtColor(self.src, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        inv = cv2.bitwise_not(mask)
        inv = cv2.medianBlur(inv, 15)
        outimage, contours, hireachy = cv2.findContours(inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for i, cnt in enumerate(contours):
            if cv2.contourArea(cnt) < 10000 or cv2.contourArea(cnt) > 5000000:
                continue
            print(cv2.contourArea(cnt))
            cv2.drawContours(inv, contours, i, (255, 255, 255), -1)
            # cv2.imshow("inv", inv)
            sorted_cnt = cnt.reshape(cnt.shape[0], cnt.shape[2])
            sorted_cnt = sorted_cnt[sorted_cnt[:, 1].argsort()]
            # print(sorted_cnt)
            # 计算轮廓中点并获取boundingbox的尺寸
            mm = cv2.moments(cnt)
            if mm['m00'] != 0:
                cx = mm['m10'] / mm['m00']  # 计算x中点
                cy = mm['m01'] / mm['m00']  # 计算y中点
                # cv2.circle(self.src, (np.int(cx), np.int(cy)), 5, (0, 0, 255), -1)
            recx, recy, recw, rech = cv2.boundingRect(cnt)

            # 查找凸点并计算香菇哪一变得凸点更多，得到香菇弯曲的朝向,并找到最低的凸点，作为下端左右分界点
            # 计数大的则向那一边凸起, 如果向右边凸起标志位为True，否则为False
            # 突起的一边采样点更多
            hull = cv2.convexHull(cnt, returnPoints=True)
            re_hull = hull.reshape(hull.shape[0], hull.shape[2])
            k = 0
            count_left = 0; count_right = 0
            for i in re_hull:
                x, y = i.ravel()
                if cy - 0.3 * rech <= y <= cy + 0.3 * rech:
                    if x <= cx:
                        count_left += 1
                    else:
                        count_right += 1
                k += 1
                cv2.circle(self.src, (x, y), 3, (255, 255, 255), -1)              # 画出凸点
                if count_left <= count_right:
                    convexflag = True
                else:
                    convexflag = False
            count_sum = count_left + count_right

            # 根据中点分别取出香菇两个边上的边界点
            sorted_left = np.zeros((1, 2), np.int8)
            sorted_right = np.zeros((1, 2), np.int8)
            for row in range(sorted_cnt.shape[0]):
                x, y = sorted_cnt[row].ravel()
                x1 = x + 5; x2 = x - 5; y1 = y + 1; y2 = y - 1
                flagx1 = inv[y, x1]; flagx2 = inv[y, x2]
                flagy1 = inv[y1, x]; flagy2 = inv[y2, x]
                if cy - 0.4 * rech <= sorted_cnt[row, 1] <= cy + 0.5 * rech:  # 在中间时不需要判断是否是上下边
                    if flagx1 == 255 and flagx2 == 0:  # 取该边界点在二值图像上左右两侧的点的至判断在左边还是右边
                        sorted_left = np.row_stack((sorted_left, sorted_cnt[row]))
                    if flagx1 == 0 and flagx2 == 255:
                        sorted_right = np.row_stack((sorted_right, sorted_cnt[row]))
                else:
                    if flagx1 == 255 and flagx2 == 0 and flagy1 == flagy2:  # 取该边界点在二值图像上左右两侧的点的至判断在左边还是右边
                        sorted_left = np.row_stack((sorted_left, sorted_cnt[row]))
                    if flagx1 == 0 and flagx2 == 255 and flagy1 == flagy2:
                        sorted_right = np.row_stack((sorted_right, sorted_cnt[row]))

            # print("vghbjnmkjnhbuy\n")
            # print(sorted_left)
            # print("gbhjnmkijnhbjnkm\n")
            # print(sorted_right)
            # 画出区分好的左右边界点   测试用
            for i in sorted_left:  # 左边点画红色
                x, y = i.ravel()
                cv2.circle(self.src, (x, y), 1, (0, 0, 255), -1)
            for i in sorted_right:
                x, y = i.ravel()
                cv2.circle(self.src, (x, y), 1, (0, 255, 0), -1)

        first = True
        if convexflag:
            right_row = 1
            for row in range(1, sorted_left.shape[0]):

                if row % 2 == 0:

                    row_dis = np.array([0, 9000], np.float32)
                    left_point_x, left_point_y = sorted_left[row].ravel()

                    for i in range(right_row, right_row + 1 + count_sum * 2):  # 寻找最小距离点的范围

                        if i >= sorted_right.shape[0]:
                            break
                        x, y = sorted_right[i].ravel()  # 遍历右边的点 并计算距离
                        dis = cal_distance(x, y, left_point_x, left_point_y)
                        # 计算距离
                        row_dis = np.row_stack((row_dis, [i, dis]))  # 将右边的index与对应的距离存储在dis中
                        if dis > 400:
                            break
                    min_dis_row = row_dis[row_dis[:, 1].argmin()]

                    if row % 50 == 0 and row != 0:              # 每50步 取一个宽度值
                        self.width.append(min_dis_row[1])

                    right_row = int(min_dis_row[0]) + 1
                    if right_row >= sorted_right.shape[0]:
                        break
                    x, y = sorted_right[right_row].ravel()
                    mid_x = int(abs(left_point_x + x) * 0.5)
                    mid_y = int(abs(left_point_y + y) * 0.5)
                    if first:
                        midpoint = np.array([mid_x, mid_y], np.int32)
                        first = False
                    midpoint = np.row_stack((midpoint, [mid_x, mid_y]))
                    cv2.line(self.src, (left_point_x, left_point_y), (x, y), (0, 255, 255), 1)

                    j = 0
                    for i in midpoint:
                        # print(i)
                        x, y = i.ravel()
                        j += 1
                        cv2.circle(self.src, (x, y), 3, (255, 0, j * 5), -1)

        else:
            left_row = 1
            for row in range(1, sorted_right.shape[0]):

                if row % 2 == 0:

                    row_dis = np.array([0, 9000], np.float32)
                    right_point_x, right_point_y = sorted_right[row].ravel()

                    for i in range(left_row, left_row + count_sum * 2):  # 寻找最小距离点的范围

                        if i >= sorted_left.shape[0]:
                            break
                        x, y = sorted_left[i].ravel()  # 遍历左边的点 并计算距离
                        dis = cal_distance(x, y, right_point_x, right_point_y)
                        # 计算距离
                        row_dis = np.row_stack((row_dis, [i, dis]))  # 将右边的index与对应的距离存储在dis中
                        if dis > 400:
                            break
                    min_dis_row = row_dis[row_dis[:, 1].argmin()]

                    if row % 50 == 0 and row != 0:              # 每50步 取一个宽度值
                        self.width.append(min_dis_row[1])

                    left_row = int(min_dis_row[0]) + 1
                    if left_row >= sorted_left.shape[0]:
                        break
                    x, y = sorted_left[left_row].ravel()
                    mid_x = int(abs(right_point_x + x) * 0.5)
                    mid_y = int(abs(right_point_y + y) * 0.5)
                    if first:
                        midpoint = np.array([mid_x, mid_y], np.int32)
                        first = False

                    midpoint = np.row_stack((midpoint, [mid_x, mid_y]))
                    cv2.line(self.src, (right_point_x, right_point_y), (x, y), (0, 255, 255), 1)
                    j = 0
                    for i in midpoint:
                        x, y = i.ravel()
                        j += 1
                        cv2.circle(self.src, (x, y), 3, (255, 0, j * 5), -1)

        self.MidAxisLength = cv2.arcLength(midpoint, False) + 0.05 * rech
        # print("中轴线长度：%.4f" % self.MidAxisLength)
        # print("宽度");print(self.width)
        # cv2.imshow("1231231", self.src)
        if self.MidAxisLength >= rech:                # 出现BUG时，中轴线距离算出超过bundingbox高度，则直接取bb*0.8长度
            self.MidAxisLength = rech * 0.8
        return self.MidAxisLength, self.width

    def cal_diameter(self):
        # 预处理
        gray = cv2.cvtColor(self.src, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # cv2.imshow("edge", binary)
        im2, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            if 50000 < cv2.contourArea(cnt) < 400000:
                cv2.drawContours(self.src, cnt, -1, (0, 255, 0), 3)
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(self.src, [box], 0, (0, 0, 255), 2)
                point1_x = box[0][0]; point1_y = box[0][1]
                point2_x = box[1][0]; point2_y = box[1][1]
                point3_x = box[2][0]; point3_y = box[2][1]
                cv2.line(self.src, (point1_x, point1_y), (point2_x,point2_y), (0, 255, 255), 2)
                cv2.line(self.src, (point2_x, point2_y), (point3_x, point3_y), (0, 255, 255), 2)
                dia1 = cal_distance(point1_x, point1_y, point2_x, point2_y)
                dia2 = cal_distance(point2_x, point2_y, point3_x, point3_y)
                self.diameter.append(dia1)
                self.diameter.append(dia2)
                print(self.diameter)

        # cv2.imshow("1", self.src)
        return self.diameter

    def cal_thickness(self):
        # 计算厚度
        # cv2.namedWindow("src", cv2.WINDOW_FREERATIO)
        # cv2.resizeWindow("src", (1000, 1000))

        gray = cv2.cvtColor(self.src, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # cv2.namedWindow("binary", cv2.WINDOW_FREERATIO)
        # cv2.resizeWindow("binary", (1000, 1000))
        # cv2.imshow("binary", binary)
        im2, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 10000 or cv2.contourArea(cnt) > 100000:
                continue
            cv2.drawContours(self.src, cnt, -1, (0, 255, 0), 3)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(self.src, [box], 0, (0, 0, 255), 2)
            point1_x = box[0][0];
            point1_y = box[0][1]
            point2_x = box[1][0];
            point2_y = box[1][1]
            point3_x = box[2][0];
            point3_y = box[2][1]
            cv2.line(self.src, (point1_x, point1_y), (point2_x, point2_y), (0, 255, 255), 2)
            cv2.line(self.src, (point2_x, point2_y), (point3_x, point3_y), (0, 255, 255), 2)
            dia1 = cal_distance(point1_x, point1_y, point2_x, point2_y)
            dia2 = cal_distance(point2_x, point2_y, point3_x, point3_y)
            print(dia1)
            print(dia2)
            if dia1 < dia2:
                self.thickness = dia1
            else:
                self.thickness = dia2
            # print(self.thickness)

        # cv2.imshow("src", self.src)
        return self.thickness



def cal_dia(argv=None):
    # 网页后端直接调用
    # 计算缓存文件夹下面图片的尺寸并保存excel文件
    support_img_file = ['jpg', 'png', 'jpeg', 'bmp', 'tif', 'gif', 'webp', 'psd']
    ws = xlwt.Workbook(encoding='utf-8')
    w = ws.add_sheet('中轴线、宽度、半径信息', cell_overwrite_ok=True)
    w.write(0, 0, '图片名')
    # w.write(0, 1, '厚度')
    w.write(0, 1, '半径1')
    w.write(0, 2, '半径2')
    w.write(0, 3, '中轴线长度')
    w.write(0, 4, '宽度1')
    w.write(0, 5, '宽度2')
    w.write(0, 6, '宽度3')
    w.write(0, 7, '宽度4')
    w.write(0, 8, '宽度5')
    w.write(0, 9, '平均宽度')
    excelname = '/home/zyf/PycharmProjects/bioweb/bioweb/cache/result.xls'
    row = 1
    for root, dirs, files in os.walk("/home/zyf/PycharmProjects/bioweb/bioweb/cache"):
        if dirs:
            # 如果有其他文件夹 就遍历这些文件夹
            for dir in dirs:
                path = os.path.join(root, dir)
                file_sorted = sorted(os.listdir(path))
                for filename in file_sorted:
                	fileformat = filename.split('.')[-1]
                	if fileformat.lower() in support_img_file: 
	                    print(filename)
	                    imgname = os.path.join(path, filename)
	                    print(imgname)
	                    try:
	                        src = image(imgname, pixelsPerCM=81, referenceLength=20)
	                        part1, part2, part3, part4 = src.image_segmentation()
	                        proce1 = ProcessMethod(part1)
	                        diameter = proce1.cal_diameter()
	                        proce2 = ProcessMethod(part2)
	                        midAxis, width = proce2.cal_MidAxis_Width()
	                        print(diameter[0] / src.pixelsPerCM)
	                    except:
	                        # 处理出错 则跳过该图像
	                        w.write(row, 0, filename)
	                        w.write(row, 1, "该图像或文件无法处理")
	                        w.write(row, 2, "可能原因：香菇尺寸过小，背景不够清晰;或者文件格式不正确")
	                        continue

	                    else:
	                        w.write(row, 0, filename)
	                        w.write(row, 1, "%.2f cm" % (diameter[0] / src.pixelsPerCM))
	                        w.write(row, 2, "%.2f cm" % (diameter[1] / src.pixelsPerCM))
	                        w.write(row, 3, "%.2f cm" % (midAxis / src.pixelsPerCM))
	                        w.write(row, 4, "%.2f cm" % (width[0] / src.pixelsPerCM))
	                        w.write(row, 5, "%.2f cm" % (width[1] / src.pixelsPerCM))
	                        if len(width) > 2:
	                            w.write(row, 6, "%.2f cm" % (width[2] / src.pixelsPerCM))
	                            if len(width) > 3:
	                                w.write(row, 7, "%.2f cm" % (width[3] / src.pixelsPerCM))
	                                if len(width) > 4:
	                                    w.write(row, 8, "%.2f cm" % (width[4] / src.pixelsPerCM))
	                        w.write(row, 9, "%.2f cm" % (np.array(width).mean() / src.pixelsPerCM))            
	                        print("\n")
	                    row += 1
        else:
            file_sorted = sorted(os.listdir("/home/zyf/PycharmProjects/bioweb/bioweb/cache"))
            for filename in file_sorted:
                print(filename)
                print(row)
                imgname = "/home/zyf/PycharmProjects/bioweb/bioweb/cache/" + filename
                fileformat = filename.split('.')[-1]
                if fileformat.lower() in support_img_file: 
	                print(imgname)
	                try:
	                    src = image(imgname, pixelsPerCM=81, referenceLength=20)
	                    part1, part2, part3, part4 = src.image_segmentation()
	                    proce1 = ProcessMethod(part1)
	                    diameter = proce1.cal_diameter()
	                    proce2 = ProcessMethod(part2)
	                    midAxis, width = proce2.cal_MidAxis_Width()
	                    print(diameter[0] / src.pixelsPerCM)
	                except:
	                    # 处理出错 则跳过该图像
	                    w.write(row, 0, filename)
	                    w.write(row, 1, "该图像无法处理")
	                    w.write(row, 2, "可能原因：香菇尺寸过小，背景不够清晰;或者文件格式不正确")
	                    continue

	                else:
	                    w.write(row, 0, filename)
	                    w.write(row, 1, "%.2f cm" % (diameter[0] / src.pixelsPerCM))
	                    w.write(row, 2, "%.2f cm" % (diameter[1] / src.pixelsPerCM))
	                    w.write(row, 3, "%.2f cm" % (midAxis / src.pixelsPerCM))
	                    w.write(row, 4, "%.2f cm" % (width[0] / src.pixelsPerCM))
	                    w.write(row, 5, "%.2f cm" % (width[1] / src.pixelsPerCM))
	                    if len(width) > 2:
	                        w.write(row, 6, "%.2f cm" % (width[2] / src.pixelsPerCM))
	                        if len(width) > 3:
	                            w.write(row, 7, "%.2f cm" % (width[3] / src.pixelsPerCM))
	                            if len(width) > 4:
	                                w.write(row, 8, "%.2f cm" % (width[4] / src.pixelsPerCM))
	                    w.write(row, 9, "%.2f cm" % (np.array(width).mean() / src.pixelsPerCM))  
	                    print("\n")
	                row += 1

    ws.save(excelname)                  # 保存excel文件

def cal_thick(pixelsPerCM):
    support_img_file = ['jpg', 'png', 'jpeg', 'bmp', 'tif', 'gif', 'webp', 'psd']
    pixelsPerCM = pixelsPerCM
    ws = xlwt.Workbook(encoding='utf-8')
    w = ws.add_sheet('半径信息', cell_overwrite_ok=True)
    w.write(0, 0, '图片名')
    w.write(0, 1, '厚度')
    excelname = '/home/zyf/PycharmProjects/bioweb/bioweb/cache/result.xls'
    row = 1

    for root, dirs, files in os.walk("/home/zyf/PycharmProjects/bioweb/bioweb/cache"):
        if dirs:
            # 如果有其他文件夹 就遍历这些文件夹
            for dir in dirs:
                path = os.path.join(root, dir)
                file_sorted = sorted(os.listdir(path))
                for filename in file_sorted:
                	fileformat = filename.split('.')[-1]
                	if fileformat.lower()  in support_img_file: 
	                    print(filename)
	                    imgname = os.path.join(path, filename)
	                    print(imgname)
	                    try:
	                        src = image(imgname, pixelsPerCM=pixelsPerCM, referenceLength=20)
	                        src = ProcessMethod(src.image)
	                        thickness = src.cal_thickness()
	                    except:
	                        w.write(row, 0, filename)
	                        w.write(row, 1, "该图像无法处理")
	                        w.write(row, 2, "可能原因：香菇尺寸过小，背景不够清晰;或者文件格式不正确")
	                    else:

	                        w.write(row, 0, filename)
	                        w.write(row, 1, "%.2f cm" % (thickness / pixelsPerCM))   # 这里需要修改 计算厚度的标尺不明
	                    row += 1
                    
        else:
            file_sorted = sorted(os.listdir("/home/zyf/PycharmProjects/bioweb/bioweb/cache"))
            for filename in file_sorted:
                print(filename)
                print(row)
                imgname = "/home/zyf/PycharmProjects/bioweb/bioweb/cache/" + filename
                fileformat = filename.split('.')[-1]
                if fileformat.lower()  in support_img_file: 
	                print(imgname)
	                try:
	                    src = image(imgname, pixelsPerCM=pixelsPerCM, referenceLength=20)
	                    src = ProcessMethod(src.image)
	                    thickness = src.cal_thickness()
	                except:
	                    w.write(row, 0, filename)
	                    w.write(row, 1, "该图像无法处理")
	                    w.write(row, 2, "可能原因：香菇尺寸过小，背景不够清晰;或者文件格式不正确")
	                else:
	                    w.write(row, 0, filename)
	                    w.write(row, 1, "%.2f cm" % (thickness / pixelsPerCM))   # 这里需要修改 计算厚度的标尺不明
	                row += 1

    ws.save(excelname)
