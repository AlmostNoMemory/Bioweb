import cv2
import numpy as np
from matplotlib import pyplot as plt

"""
==========================================
Pyhton OpenCV 笔记 
==========================================
"""

"""
===================================
Core Operations 基本操作
===================================
"""

""" ----- 文件读取和保存 -----
"""
# 读取图片
src = 'file'                                 # 可以是绝对路径，也可以是相对路径
image = cv2.imread(src)
src2 = '/dir/dir/filename'
image2 = cv2.imwrite(src2)

# 读取视频帧
capture = cv2.VideoCapture
while True:
    ret, frame = capture.read()             # .read方法有一个返回值ret为0时说明视频结束
    if not ret:                             # ret为0时结束读取
        break
    cv2.imshow(frame)                       # 显示一帧画面
    c = cv2.waitKey(40)                     # 等待延迟
    if c == 27:                             # 当按下Esc时退出显示画面
        break

# 保存图片
savename = ''                               # 可以包含路径
cv2.imwrite(savename, image)


""" ------ numpy数组操作 -----
"""
array = np.zeros([3, 3, 3], dtype=np.uint32)   # 创建一个3维0矩阵 数据类型为 uint32 类型根据实际需求改变

array = np.ones([3, 3, 3], dtype=np.uint32)    # 创建一个3维元素都是1的矩阵
array[:, :, 0] = np.ones([3, 3]) * 255         # 给array第三维赋值

matrix = array.reshape([1, 9])                 # 改变矩阵形状


min_dis_row = row_dis[row_dis[:, 1].argmin()]    # 取出axis=1轴上之最大的一组
sorted_point = reshp[reshp[:, 1].argsort()]        # 按axis=1上的值排序

""" ----- 色彩空间 -----
"""
# opencv默认的色彩空间是BGR -- BGR取值范围都是0-255
# 其他色彩空间： HSV -- h：0-180; s:0-255; v:0-255;
#               （为什么是0-180,而不是0-360？归一化问题，0-180使用int8就可以表示,0-360会溢出）
#             HIS; YCrCb; YUV -- YUV是linux或安卓原始格式

# 色彩空间转换
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)       # 转化为灰度
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)         # 转化为hsv 其他同理

# 像素取反操作
inverse = cv2.bitwise_not(image)

# mask操作
mask = cv2.inRange(hsv, lowerb=[], upperb=[])         # 通过mask可以提取在range范围内的像素点 最好使用hsv图

# 通道分离与合并
b, g, r = cv2.split(src)
cv2.merge([b, g, r])



""" ----- 像素运算 ------
"""
# 加减乘除
add = cv2.add(image, image2)
sub = cv2.subtract(image, image2)
mul = cv2.multiply(image, image2)
div = cv2.divide(image, image2)

# 求平均值和方差
mean, var = cv2.meanStdDev(image)
print(mean)                                 # 均值比较低说图片明比较暗, 方差较大说明像素之间差距较大
#                                               方差较小说明图片信息量比较少

# 逻辑运算 与或非
cv2.bitwise_not(image, image2)
cv2.bitwise_and()                           # 与运算 原图和mask与运算可以加覆盖上mask
cv2.bitwise_or()                            # 或运算

# eg： 调整亮度和对比度
def contrast_brighness(image, b, c):
    h, w, ch = image.shapeb
    blank = np.zeros([h, w, ch], image.dtype)
    dst = cv2.addWeighted(image, c, blank, 1-c, b)
    cv2.imshow("contrast", dst)


contrast_brighness(image, 1.2, 10)          # 调用函数， 第二个参数为亮度值，第三个为对比度值
# 调整亮度实际上就是每个像素点加一定的值，这样像素点的值变高往[255,255,255],显得更亮（更白）
#    调整对比度实际上就是每个像素的乘一定的倍数，这样像素点之间的差异就会变大




"""
===================================
Image Processing 图像处理
===================================
"""

""" ----- 图像二值化 -----
"""
# 全局阈值
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
print(ret)                                  # 阈值
cv2.imshow("binary", binary)
# cv2.THRESH_OTSU -- 当图像有两个直方图波峰时效果好  cv2.THRESH_TRIANGLE -- 三角阈值计算法,有单个直方图波峰时效果较好
# 23项可以自己手动设置阈值，但是后面的方法需要去掉

# 局部二值化
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=, C=10)
# 可以保留轮廓
# blockSize -- 考虑的临近像素的大小，必须为奇数
# C -- 是一个偏移量的调整值

# 超大图像二值化  首先对图像进行分割



""" ----- 模糊操作 -----
"""
# 模糊操作的基本原理：基于离散卷积
# 定义好每个卷积核，不同的核得到不同的效果 卷积核一般为奇数

# 均值模糊
cv2.blur(image, (5,5))                           # 第二个参数为卷积核的size -- (宽，长)
# 用kernel中像素的平均值替换中心pixel的像素值
# 对随机噪声有很好的去噪效果

# 中值模糊
cv2.medianBlur(image, (5,5))
# 用kernel中像素的中值替换中心pixel的像素值
# 对椒盐噪声有很好的去噪效果

# 自定义卷积
cv2.filter2D(image, -1, kernel= kernal, anchor= )                    # kernal为自定义卷积核，anchor为卷积铆点
# 可以自定义卷积核 eg:
kernal = np.ones([5,5], np.float32) / 25                           # 该kernal与中值模糊效果相同
kernal = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], np.float32)       # 该kernal可以实现锐化
# 需要注意自定义kernal： 需要是奇数维数，相加小于等于1,若大于1需要除以一个数保证小于等于一（归一化）


# 高斯模糊
cv2.GaussianBlur(src, ksize=3, sigmaX=0)                  # sigma是在x或y方向上的标准偏差
# 对高斯噪声有很好的去噪效果
# 二维高斯模糊处理在计算过程中一般拆分成两个一维，可以减少计算量
# eg： 3*3 需要9次乘法，一次除法   拆分成两次1*3,需要6次乘法，两次除法


# 边缘保留滤波
# gi：边缘像素差异较大，不去滤波和模糊，就可以起到保留边缘的作用

# 高斯双边模糊
cv2.bilateralFilter(src, d=0, sigmaColor=100, sigmaSpace=15)             # 后面

# 均值迁移模糊
cv2.pyrMeanShiftFiltering()
# 边缘会出现过度模糊的现象，类似油画的效果



""" ----- 图像形态学 -----
"""
# 结构元素
kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(3,3))
# 如果提取横向水平的结构元素，kernel可以设置为（15,1）
# 同理，竖向的结构元素为（1,15）

# 膨胀
cv2.dilate(image, kernel=,)
# 用kernal内最大像素的值替代中点的值，也可以看成一个最大值滤波
# 膨胀的作用：  对象大小增加一个像素
#             平滑对象边缘
#             减少或者填充对象之间的距离

# 腐蚀
cv2.erode(image, kernel=,)
# 用kernal的最小像素值替代中点像素的值，也就是一个最小值滤波
# 腐蚀的作用：  对象大小减少一个像素
#             平滑对象边缘
#             弱化或者分割对象之间的半岛型连接

# 开操作 open
cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
# 基于膨胀与腐蚀操作组合形成的， 主要应用与二值图像分析
# 开操作 = 腐蚀 + 膨胀 ，输入图像 + 结构元素
# 可以消除图像内的噪点

# 闭操作 close
cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
# 与开操作相反
# 闭操作 = 膨胀 + 腐蚀 ，输入图像 + 结构元素
# 填充小的封闭区域

# 顶帽 tophat
# 原图像与开操作之间的差值图像

# 黑帽 blackhat
# 原图像与必操作之间的差值图像

# 形态学梯度
gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel=)

# 结构元素
kernel = cv2.getStructuringElement(shape=cv2.MORPH_GRADIENT, ksize=(x, y))
# 获取特定形状的结构元素



""" ----- 图像梯度 -----
"""
# 一阶导数 变化最大的地方就是图像边缘
cv2.Sobel(image, ddepth=cv2.CV_32F,dx=,dy=)                      # Sobel算子
# 如果选取取dx=1,dy=0则只计算x方向上的梯度，也即边缘

cv2.Scharr()                                                     # Scharr算子
# Scharr算子是sobel算子的增强版本，但是对噪声更加敏感

# 二阶导数 最大变化处的值为零即边缘是零值，根据此理论就可以提取边缘
cv2.Laplacian(image, ddepth=cv2.CV_32F)                          # 拉普拉斯算子

# 使用filter2D自定义拉普拉斯kernal所得结果也是一样的
kernal1 = np.array([[0, 1, 0],[1, -4, 1], [0, 1, 0]])             # 四领域拉普拉斯算子
dst = cv2.filter2D(image, ddepth=cv2.CV_32F, kernel=kernal)
lpls = cv2.convertScaleAbs(dst)                                   # 需要取绝对值



""" ----- Canny边缘检测 -----
"""
edge = cv2.Canny()



""" ----- 轮廓发现 -----
"""
cloneImage, contours, heriachy = \
    cv2.findContours(image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
#  返回值  cloneImage -- 经过处理的图片
#         contours -- 提取到的轮廓
#         heiachy  -- 层次信息
#  输入的图像可以是edge边缘提取的图像，也可以是二值化后的图像
# method 如果选择SIMPLE即选择一些采样点，可以节省内存，在图像边缘规则时比较好用， NONE是选择所有的点

# 轮廓信息
M = cv2.moments(contours[i])
# M中包含了一些轮廓的信息
# eg：中心点
center_x = int(M['m10']/M['m00'])
center_y = int(M['m01']/M['m00'])

# 遍历轮廓
for i, contour in enumerate(contours):
    if cv2.contourArea(contour) < 2000:
        continue
    else:
        cv2.drawContours(image, contours, i, (0, 255, 255), 1)
        print(i)



""" ----- 图像直方图 -----
"""
# bin的大小 = (图像中不同象素值的个数)/(Bin的数目)

# 绘制三通道直方图
def image_hist(image):
    """图像直方图"""
    color = ('blue', 'green', 'red')
    for i ,color in enumerate(color):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 255])
    plt.show()
# 可以直观的得到图像的特征


# 直方图均衡化
cv2.equalizeHist(image)
# 可以增强对比度，但是只能对灰度图做处理


# 局部直方图均衡化
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
dst = clahe.apply(image)
# 只能处理单通道的图片


# 直方图反向投影图
def back_projection(image):
    """直方图反向投影"""
    sample = cv2.imread('target_name')
    target = cv2.imread('sample_name')
    roi_hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
    target_hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)

    cv2.imshow("sample", sample)
    cv2.imshow("target", target)

    roiHist = cv2.calcHist(roi_hsv, [0, 1], None, [180,256], [0, 180, 0, 256])
    # cv2.calcHist(images=, channels=, mask=, histSize=, ranges=)
    # histSize -- 直方图横坐标的区间数，如果为10,则横坐标分为10份，然后计算每个区间的像素点总和
    # ranges -- 指出每个区间的范围
    cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)
    dst = cv2.calcBackProject(target_hsv, [0, 1], roiHist, [0, 180, 0, 256], 1)
    cv2.imshow("BackProjection", dst)
# 可以实现从target中提取出直方图与sample类似的区域，类似于扣图


""" ----- 傅利叶变换 -----
"""
# 可以将图片看成是x、y方向上的分别采样的信号，所以可以做频域分析
# 振幅在短时间内变化很快时，可以看作是一个高频信号，所以对于图像，在边缘或者噪点处是高频区域




""" -----直线检测 圆检测 -----    # 未学习
"""
# 霍夫直线变换  原理：极坐标变换
# 霍夫圆变换



""" ----- 分水岭算法 -----
"""
# 距离变换
cv2.distanceTransform(src, cv2.DIST_L1, 3)
# 计算图像中每一个非零点距离离自己最近的零点的距离




""" ----- ROI与泛洪填充 ------
"""
# ROI reign of interest 感兴趣区域：从被处理的图像以方框、圆、椭圆、不规则多边形等方式勾勒出需要处理的区域
ROI = src[20:30, 100:300]           # 长，宽上的取值

# eg 将ROI区域变为灰色
def ROI_demo():
    src = 'dir/filename'
    face = src[200:400, 150:400]                            # 选取ROI区域
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)           # 转化为灰度图
    backface = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)       # 转化回BGR， 不然会缺少一个channel
    src[200:400, 150:400] = backface                        # 将ROI插回原图
    cv2.imshow("face", face)

# eg 泛洪填充颜色
def fill_color_demo(image):
    h, w = image.shape[:2]
    mask = np.zeros([h+2, w+2], np.uint8)                   # mask必须为单通道8位
    cv2.floodFill(image, mask, seedPoint=, newVal=, loDiff=, upDiff=, cv2.FLOODFILL_FIXED_RANGE)
    # seedPoint为开始填充的点，newVal为填充的颜色，loDiff\upDiff为需要填充的最低值和最高值，最后为填充模式
    # cv2.FLOODFILL_FIXED_RANGE -- 改变图像，泛洪填充
    # cv2.FLOODFILL_MASK_ONLY -- 不改变图像，只填充mask本身，忽略新的颜色值参数，mask值必须为0




""" ----- 模板匹配 -----
"""
# 模板匹配就是在整个图像区域发现与给定的子图像匹配的小块区域
# 需要一个模板图像T，然后需要原图像S
# 从上到下，从左到右

# 度量相似程度
# cv2.TM_SQDIFF -- 平方不同



""" ----- 图像金字塔 -----
"""
# eg reduce金字塔
def down_demo(image):
    level = 3
    temp = image.copy()
    pyramid_image = []
    for i in range(level):
        dst = cv2.pyrDown(temp)
        pyramid_image.append(dst)
        cv2.imshow("pyramid_demo_"+str(i), dst)
        temp = dst.copy()
    return pyramid_image



""" ----- 对象测量 -----
"""
# 弧长与面积
# 首先进行轮廓发现  轮廓是一系列的点，然后就可以计算每个轮廓的弧长与面积的像素值

# eg
def measire(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (7, 7), 0)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    print("threshold value: %s"%ret)
    # cv2.imshow("binary", binary)
    outimage, contours, hireachy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area < 2000) or (area > 100000):
            continue
        x, y, w, h = cv2.boundingRect(contour)
        mm = cv2.moments(contour)
        type(mm)
        #print(mm)
        if mm['m00'] != 0:
            cx = mm['m10']/mm['m00']        # 计算x中点
            cy = mm['m01']/mm['m00']        # 计算y中点
            cv2.circle(image, (np.int(cx),np.int(cy)), 3, (0, 255, 255), -1)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        print("contour area %s"%area)
        approxCure = cv2.approxPolyDP(contour, epsilon=4, closed=False)
        # epsilon 决定曲线与边界的距离     close 是否封闭
        #print(approxCure)
        print(approxCure.shape)
        cv2.drawContours(image, contours, i, (0, 255, 0), 2)
        # 画出拟合的多边形  approxCure的第一个维度就是拟合的曲线数量
    cv2.imshow("1", image)




"""
===============================================
Feature Detection and  Description 特征发现与描述
===============================================
"""
# 寻找特征，直觉的方式就是寻找附近区域中有最多变量的区域
# 例如物体的角（corner）就是一个好的特征，因为可以很容易在原图中找到这个特征的准确位置
# 将发现的特征描述出来，就可以在其他图片中根据该描述寻找、匹配特征

""" ----- 哈里斯角检测 -----
"""
dst = cv2.cornerHarris(gray, 2, 21, 0.04)


# Shi-Tomasi特征寻找
corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)




