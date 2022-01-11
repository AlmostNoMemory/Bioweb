from django.shortcuts import render
from django.http import HttpResponse,FileResponse
from .models import IMG
from .forms import FileFieldForm,UploadFileForm
from django.http import HttpResponseRedirect, Http404, JsonResponse
from django.urls import reverse
from django.views.generic.edit import FormView
from django.contrib.auth.decorators import login_required
import os
import xlwt
import cv2
import numpy as np
import time
from . import bio_ClassAndFunction
import zipfile
import rarfile
import shutil
import sys

def index(request):
    # 主页
    return render(request, 'mainsite/index.html')

@login_required
def calculation(request):
    # 显示两种尺寸计算入口的界面

    return render(request, 'mainsite/calculation.html')

@login_required
def cal_dia_centerline_width(request):
    # 计算中轴线、半径和宽度的界面

    # 清除cache目录下的文件
    cache_path = "/home/zyf/project/bioweb/bioweb/cache/"
    shutil.rmtree(cache_path)
    # 创建新的cache文件夹
    os.mkdir(cache_path)

    if request.method != 'POST':
        form = FileFieldForm()
        print(form)
        print('\n')
    else:
        form = FileFieldForm(request.POST, request.FILES)
        files = request.FILES.getlist('file_field')
        print(form)
        if form.is_valid:
            for f in files:
                destination = open(os.path.join("/home/zyf/project/bioweb/bioweb/cache",str(f)),'wb+')
                for chunk in f.chunks():
                    destination.write(chunk)
                destination.close()           # 将上传的文件写入到本地
            print("写入到本地--完成")
            bio_ClassAndFunction.cal_dia()    # 调用处理函数
            print("处理完成")


        return HttpResponseRedirect(reverse('mainsite:success'))

    return render(request, 'mainsite/cal_dia_centerline_width.html')


@login_required
def cal_dia_centerline_width_compress_fake(request):
    return render(request, 'mainsite/cal_dia_compress.html')

@login_required
def cal_dia_centerline_width_compress(request):
    # 计算中轴线、半径和宽度的界面  压缩包处理界面
    print('begin to calculate')
    # 计算厚度  压缩包处理界面
    global num_progress
    global eta_time
    global imgshow
    num_progress = 0
    imgshow = None
    last_file = None
    last_2_file = None
    last_3_file = None
    eta_time = 0
    i = 0
    filenum = 0
    for root, dirs, files in os.walk("/home/zyf/project/bioweb/bioweb/cache"):
        if dirs:
            for dir in dirs:
                path = os.path.join(root, dir)
                filenum += len(os.listdir(path))
        else:
            filenum = len(files)

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
    excelname = '/home/zyf/project/bioweb/bioweb/cache/result.xls'
    row = 1
    for root, dirs, files in os.walk("/home/zyf/project/bioweb/bioweb/cache"):
        if dirs:
            # 如果有其他文件夹 就遍历这些文件夹
            for dir in dirs:
                path = os.path.join(root, dir)
                file_sorted = sorted(os.listdir(path))
                for filename in file_sorted:
                    fileformat = filename.split('.')[-1]
                    if fileformat.lower() in support_img_file:
                        start = time.time()
                        print(filename)
                        imgname = os.path.join(path, filename)
                        print(imgname)
                        try:
                            src = bio_ClassAndFunction.image(imgname, pixelsPerCM=81, referenceLength=20)
                            part1, part2, part3, part4 = src.image_segmentation()
                            proce1 = bio_ClassAndFunction.ProcessMethod(part1)
                            diameter = proce1.cal_diameter(filename)
                            proce2 = bio_ClassAndFunction.ProcessMethod(part2)
                            midAxis, width = proce2.cal_MidAxis_Width(filename)
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
                        i += 1
                        imgshow = last_file
                        num_progress = format(i / filenum * 100, '.1f')
                        eta_time = int((time.time() - start) * (filenum - i))

                        delfile = last_3_file
                        last_3_file = last_2_file
                        last_2_file = last_file
                        last_file = filename
                        print('last two file', last_2_file)
                        print('last file ', last_file)
                        print('file ', filename)
                        # del last file
                        if delfile:
                            delroot = "/home/zyf/project/bioweb/bioweb/static/vis/cache"
                            delfile_path1 = os.path.join(delroot, "mid" + delfile)
                            os.remove(delfile_path1)
                            delfile_path2 = os.path.join(delroot, "dia" + delfile)
                            os.remove(delfile_path2)

        else:
            file_sorted = sorted(os.listdir("/home/zyf/project/bioweb/bioweb/cache"))
            for filename in file_sorted:
                print(filename)
                print(row)
                imgname = "/home/zyf/project/bioweb/bioweb/cache/" + filename
                fileformat = filename.split('.')[-1]
                if fileformat.lower() in support_img_file:
                    start = time.time()
                    print(imgname)
                    try:
                        src = bio_ClassAndFunction.image(imgname, pixelsPerCM=81, referenceLength=20)
                        part1, part2, part3, part4 = src.image_segmentation()
                        proce1 = bio_ClassAndFunction.ProcessMethod(part1)
                        diameter = proce1.cal_diameter(filename)
                        proce2 = bio_ClassAndFunction.ProcessMethod(part2)
                        midAxis, width = proce2.cal_MidAxis_Width(filename)
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
                    i += 1
                    imgshow = last_file
                    num_progress = format(i / filenum * 100, '.1f')
                    eta_time = int((time.time() - start) * (filenum - i))

                    delfile = last_3_file
                    last_3_file = last_2_file
                    last_2_file = last_file
                    last_file = filename
                    print('last two file', last_2_file)
                    print('last file ', last_file)
                    print('file ', filename)
                    # del last file
                    if delfile:
                        delroot = "/home/zyf/project/bioweb/bioweb/static/vis/cache"
                        delfile_path1 = os.path.join(delroot, "mid" + delfile)
                        os.remove(delfile_path1)
                        delfile_path2 = os.path.join(delroot, "dia" + delfile)
                        os.remove(delfile_path2)

    ws.save(excelname)  # 保存excel文件

    res = 100
    return JsonResponse(res, safe=False)


@login_required
def cal_thickness(request):

    # 清除cache目录下的文件
    cache_path = "/home/zyf/project/bioweb/bioweb/cache/"
    shutil.rmtree(cache_path)
    # 创建新的cache文件夹
    os.mkdir(cache_path)

    # 计算厚度的界面
    if request.method != 'POST':
        form = FileFieldForm()
        print(form)
    else:
        form = FileFieldForm(request.POST, request.FILES)
        files = request.FILES.getlist('file_field')
        print(files)
        if form.is_valid:
            for f in files:
                #print(f)
                destination = open(os.path.join("/home/zyf/project/bioweb/bioweb/cache",str(f)),'wb+')
                for chunk in f.chunks():
                    destination.write(chunk)
                destination.close()
                print("写入到本地--完成")
            bio_ClassAndFunction.cal_thick(99)              # 这里需要填入计算厚度的参考长度pixelsPerCM
            print("处理完成")


        return HttpResponseRedirect(reverse('mainsite:success'))

    return render(request, 'mainsite/cal_thick.html')

@login_required
def cal_thickness_compress_fake(request):
    return render(request, 'mainsite/cal_thick_compress.html')

@login_required
def cal_thickness_compress(request):

    print('begin to calculate')
    # 计算厚度  压缩包处理界面
    global num_progress
    global eta_time
    global imgshow
    last_file = None
    last_2_file = None
    last_3_file = None
    imgshow = None
    num_progress = 0
    eta_time = 0
    support_img_file = ['jpg', 'png', 'jpeg', 'bmp', 'tif', 'gif', 'webp', 'psd']
    pixelsPerCM = 99
    # pixelsPerCM = pixelsPerCM
    ws = xlwt.Workbook(encoding='utf-8')
    w = ws.add_sheet('半径信息', cell_overwrite_ok=True)
    w.write(0, 0, '图片名')
    w.write(0, 1, '厚度')
    excelname = '/home/zyf/project/bioweb/bioweb/cache/result.xls'
    row = 1
    i = 0

    filenum = 0
    for root, dirs, files in os.walk("/home/zyf/project/bioweb/bioweb/cache"):
        if dirs:
            for dir in dirs:
                path = os.path.join(root, dir)
                filenum += len(os.listdir(path))
        else:
            filenum = len(files)

    for root, dirs, files in os.walk("/home/zyf/project/bioweb/bioweb/cache"):
        if dirs:
            # 如果有其他文件夹 就遍历这些文件夹
            for dir in dirs:
                path = os.path.join(root, dir)
                file_sorted = sorted(os.listdir(path))

                for filename in file_sorted:
                    fileformat = filename.split('.')[-1]

                    if fileformat.lower() in support_img_file:
                        start = time.time()
                        print(filename)
                        imgname = os.path.join(path, filename)
                        print(imgname)
                        try:
                            src = bio_ClassAndFunction.image(imgname, pixelsPerCM=pixelsPerCM, referenceLength=20)
                            src = bio_ClassAndFunction.ProcessMethod(src.image)
                            thickness = src.cal_thickness(filename)
                        except:
                            w.write(row, 0, filename)
                            w.write(row, 1, "该图像无法处理")
                            w.write(row, 2, "可能原因：香菇尺寸过小，背景不够清晰;或者文件格式不正确")
                        else:

                            w.write(row, 0, filename)
                            w.write(row, 1, "%.2f cm" % (thickness / pixelsPerCM))  # 这里需要修改 计算厚度的标尺不明
                        row += 1

                        i += 1
                        num_progress = format(i / filenum * 100, '.1f')
                        eta_time = int((time.time() - start) * (filenum - i))
                        imgshow = last_file

                        delfile = last_3_file
                        last_3_file = last_2_file
                        last_2_file = last_file
                        last_file = filename
                        print('last two file', last_2_file)
                        print('last file ', last_file)
                        print('file ', filename)
                        # del last file
                        if delfile:
                            delroot = "/home/zyf/project/bioweb/bioweb/static/vis/cache"
                            delfile_path = os.path.join(delroot, delfile)
                            os.remove(delfile_path)


        else:
            file_sorted = sorted(os.listdir("/home/zyf/project/bioweb/bioweb/cache"))
            for filename in file_sorted:

                print(filename)
                print(row)
                imgname = "/home/zyf/project/bioweb/bioweb/cache/" + filename
                fileformat = filename.split('.')[-1]
                if fileformat.lower() in support_img_file:
                    start = time.time()
                    print(imgname)
                    try:
                        src = bio_ClassAndFunction.image(imgname, pixelsPerCM=pixelsPerCM, referenceLength=20)
                        src = bio_ClassAndFunction.ProcessMethod(src.image)
                        thickness = src.cal_thickness(filename)
                    except:
                        w.write(row, 0, filename)
                        w.write(row, 1, "该图像无法处理")
                        w.write(row, 2, "可能原因：香菇尺寸过小，背景不够清晰;或者文件格式不正确")
                    else:
                        w.write(row, 0, filename)
                        w.write(row, 1, "%.2f cm" % (thickness / pixelsPerCM))  # 这里需要修改 计算厚度的标尺不明
                    row += 1
                i += 1
                print("i=", i)
                print('filenums=,', filenum)
                num_progress = format(i / filenum * 100, '.1f')
                eta_time = int((time.time() - start) * (filenum - i))
                imgshow = last_file

                delfile = last_3_file
                last_3_file = last_2_file
                last_2_file = last_file
                last_file = filename
                print('last two file', last_2_file)
                print('last file ', last_file)
                print('file ', filename)
                # del last file
                if delfile:
                    delroot = "/home/zyf/project/bioweb/bioweb/static/vis/cache"
                    delfile_path = os.path.join(delroot, delfile)
                    os.remove(delfile_path)

    ws.save(excelname)

   # 调用处理函数
    print("处理完成")
    res = 100
    return JsonResponse(res, safe=False)


@login_required
def success(request):
    # 上传成功的界面
    return render(request, 'mainsite/success.html')

@login_required
def success_download_excel(request):
    # 下载excel文件的响应
    filename = "/home/zyf/project/bioweb/bioweb/cache/result.xls"
    the_file_name = 'result.xls'
    file = open(filename, 'rb')
    response = FileResponse(file)
    response['Content-Type'] = 'application/octet-stream'
    response['Content-Disposition'] = 'attachment;filename="{0}"'.format(the_file_name)
    return response


@login_required
def up_load_compressfile_1(request):
    global progress
    global status
    # 清除cache目录下的文件
    cache_path = "/home/zyf/project/bioweb/bioweb/compress_cache"
    shutil.rmtree(cache_path)
    # 创建新的cache文件夹
    os.mkdir(cache_path)
    progress = 5
    status = "文件上传中..."

    if request.method != 'POST':
        form = FileFieldForm()
        print(form)
        print('\n')
    else:
        status = "文件上传中..."
        progress = 10
        form = FileFieldForm(request.POST, request.FILES)
        files = request.FILES.getlist('file_field')
        size = request.FILES['file_field'].size
        print(form)
        if form.is_valid:
            for f in files:
                destination = open(os.path.join("/home/zyf/project/bioweb/bioweb/compress_cache", str(f)), 'wb+')
                size_write = 0
                for chunk in f.chunks():
                    size_write += sys.getsizeof(chunk)
                    progress = (size_write / size) * 100 * (4 / 10) + 10
                    destination.write(chunk)
                destination.close()  # 将上传的文件写入到本地

            print("写入到本地--完成")
            print(progress)
            status = "后台文件解压中..."

            # 解压操作
            file_sorted = sorted(os.listdir("/home/zyf/project/bioweb/bioweb/compress_cache"))
            for filename in file_sorted:
                print(filename)
                format = str(filename).split('.')[-1]
                print(format)
                # if format != 'zip' or format != 'rar':
                if format == 'zip':
                    target = "/home/zyf/project/bioweb/bioweb/compress_cache/" + filename
                    zipf = zipfile.ZipFile(target, 'r')
                    n = len(zipf.namelist())
                    i = 0
                    for file in zipf.namelist():  # f.namelist()返回列表，列表中的元素为压缩文件中的每个文件
                        progress = 50 + (i / n) * 100 * 0.5
                        i += 1
                        print(progress)
                        zipf.extract(file, "/home/zyf/project/bioweb/bioweb/cache/")
                    print("zip文件解压完成")

                if format == 'rar':
                    target = "/home/zyf/project/bioweb/bioweb/compress_cache/" + filename
                    rarf = rarfile.RarFile(target)
                    n = len(rarf.namelist())
                    i = 0
                    for file in rarf.namelist():
                        rarf.extract(file, "/home/zyf/project/bioweb/bioweb/cache/")
                        progress = 50 + (i / n) * 100 * 0.5
                        i += 1
                        print(progress)
                    print("rar文件解压完成")
            return HttpResponseRedirect(reverse('mainsite:cal1compress'))
    return render(request, 'mainsite/upload.html')

@login_required
def up_load_compressfile_2(request):
    global progress
    global status
    # 清除cache目录下的文件
    cache_path = "/home/zyf/project/bioweb/bioweb/compress_cache"
    shutil.rmtree(cache_path)
    # 创建新的cache文件夹
    os.mkdir(cache_path)
    progress = 5
    status = "文件上传中..."

    if request.method != 'POST':
        form = FileFieldForm()
        print(form)
        print('\n')
    else:
        status = "文件上传中..."
        progress = 10
        form = FileFieldForm(request.POST, request.FILES)
        files = request.FILES.getlist('file_field')
        size = request.FILES['file_field'].size
        print(form)
        if form.is_valid:
            for f in files:
                destination = open(os.path.join("/home/zyf/project/bioweb/bioweb/compress_cache", str(f)), 'wb+')
                size_write = 0
                for chunk in f.chunks():
                    size_write += sys.getsizeof(chunk)
                    progress = (size_write / size) * 100 * (4 / 10) + 10
                    destination.write(chunk)
                destination.close()  # 将上传的文件写入到本地

            print("写入到本地--完成")
            print(progress)
            status = "后台文件解压中..."

            # 解压操作
            file_sorted = sorted(os.listdir("/home/zyf/project/bioweb/bioweb/compress_cache"))
            for filename in file_sorted:
                print(filename)
                format = str(filename).split('.')[-1]
                print(format)
                # if format != 'zip' or format != 'rar':
                if format == 'zip':
                    target = "/home/zyf/project/bioweb/bioweb/compress_cache/" + filename
                    zipf = zipfile.ZipFile(target, 'r')
                    n = len(zipf.namelist())
                    i = 0
                    for file in zipf.namelist():  # f.namelist()返回列表，列表中的元素为压缩文件中的每个文件
                        progress = 50 + (i / n) * 100 * 0.5
                        i += 1
                        print(progress)
                        zipf.extract(file, "/home/zyf/project/bioweb/bioweb/cache/")
                    print("zip文件解压完成")

                if format == 'rar':
                    target = "/home/zyf/project/bioweb/bioweb/compress_cache/" + filename
                    rarf = rarfile.RarFile(target)
                    n = len(rarf.namelist())
                    i = 0
                    for file in rarf.namelist():
                        rarf.extract(file, "/home/zyf/project/bioweb/bioweb/cache/")
                        progress = 50 + (i / n) * 100 * 0.5
                        i += 1
                        print(progress)
                    print("rar文件解压完成")
            return HttpResponseRedirect(reverse('mainsite:cal2compress'))
    return render(request, 'mainsite/upload.html')


def show_progress(request):
    print("num progress=", num_progress)
    print("eta_time=", eta_time)
    return JsonResponse([num_progress, eta_time, imgshow], safe=False)


def show_uploade_progress(request):
    # progress = 0
    return JsonResponse([format(progress, '.1f'), status], safe=False)

# Create your views here.
