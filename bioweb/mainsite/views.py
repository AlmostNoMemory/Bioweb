from django.shortcuts import render
from django.http import HttpResponse,FileResponse
from .models import IMG
from .forms import FileFieldForm,UploadFileForm
from django.http import HttpResponseRedirect,Http404
from django.urls import reverse
from django.views.generic.edit import FormView
from django.contrib.auth.decorators import login_required
import os
import xlwt
import cv2
import numpy as np
from . import bio_ClassAndFunction
import zipfile
import rarfile
import shutil

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
    cache_path = "/home/zyf/PycharmProjects/bioweb/bioweb/cache/"
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
                destination = open(os.path.join("/home/zyf/PycharmProjects/bioweb/bioweb/cache",str(f)),'wb+')
                for chunk in f.chunks():
                    destination.write(chunk)
                destination.close()           # 将上传的文件写入到本地
            print("写入到本地--完成")
            bio_ClassAndFunction.cal_dia()    # 调用处理函数
            print("处理完成")


        return HttpResponseRedirect(reverse('mainsite:success'))

    return render(request, 'mainsite/cal_dia_centerline_width.html')



@login_required
def cal_dia_centerline_width_compress(request):
    # 计算中轴线、半径和宽度的界面  压缩包处理界面

    # 清除cache目录下的文件
    cache_path = "/home/zyf/PycharmProjects/bioweb/bioweb/compress_cache"
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
                destination = open(os.path.join("/home/zyf/PycharmProjects/bioweb/bioweb/compress_cache",str(f)),'wb+')
                for chunk in f.chunks():
                    destination.write(chunk)
                destination.close()           # 将上传的文件写入到本地
            print("写入到本地--完成")
            # 解压操作
            file_sorted = sorted(os.listdir("/home/zyf/PycharmProjects/bioweb/bioweb/compress_cache"))
            for filename in file_sorted:
                print(filename)
                format = str(filename).split('.')[-1]
                print(format)
                #if format != 'zip' or format != 'rar':
                if format == 'zip':
                    target =  "/home/zyf/PycharmProjects/bioweb/bioweb/compress_cache/" + filename
                    zipf = zipfile.ZipFile(target,'r')
                    for file in zipf.namelist(): #f.namelist()返回列表，列表中的元素为压缩文件中的每个文件
                        zipf.extract(file,"/home/zyf/PycharmProjects/bioweb/bioweb/cache/")
                    print("zip解压完成")
                if format == 'rar':
                    target =  "/home/zyf/PycharmProjects/bioweb/bioweb/compress_cache/" + filename
                    rarf = rarfile.RarFile(target)
                    for file in rarf.namelist():
                        rarf.extract(file,"/home/zyf/PycharmProjects/bioweb/bioweb/cache/")
                    #rarf.extractall("/home/zyf/PycharmProjects/bioweb/bioweb/cache/")

            bio_ClassAndFunction.cal_dia()    # 调用处理函数
            print("处理完成")


        return HttpResponseRedirect(reverse('mainsite:success'))

    return render(request, 'mainsite/cal_dia_compress.html')


@login_required
def cal_thickness(request):

    # 清除cache目录下的文件
    cache_path = "/home/zyf/PycharmProjects/bioweb/bioweb/cache/"
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
                destination = open(os.path.join("/home/zyf/PycharmProjects/bioweb/bioweb/cache",str(f)),'wb+')
                for chunk in f.chunks():
                    destination.write(chunk)
                destination.close()
                print("写入到本地--完成")
            bio_ClassAndFunction.cal_thick(99)              # 这里需要填入计算厚度的参考长度pixelsPerCM
            print("处理完成")


        return HttpResponseRedirect(reverse('mainsite:success'))

    return render(request, 'mainsite/cal_thick.html')



@login_required
def cal_thickness_compress(request):
    # 计算厚度  压缩包处理界面

    # 删除就的cache文件夹
    cache_path = "/home/zyf/PycharmProjects/bioweb/bioweb/compress_cache"
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
                destination = open(os.path.join("/home/zyf/PycharmProjects/bioweb/bioweb/compress_cache",str(f)),'wb+')
                for chunk in f.chunks():
                    destination.write(chunk)
                destination.close()           # 将上传的文件写入到本地
            print("写入到本地--完成")
            # 解压操作
            file_sorted = sorted(os.listdir("/home/zyf/PycharmProjects/bioweb/bioweb/compress_cache"))
            for filename in file_sorted:
                format = str(filename).split('.')[-1]
                print(format)
                target =  "/home/zyf/PycharmProjects/bioweb/bioweb/compress_cache/" + filename
                f = zipfile.ZipFile(target,'r')
                for file in f.namelist(): #f.namelist()返回列表，列表中的元素为压缩文件中的每个文件
                    f.extract(file,"/home/zyf/PycharmProjects/bioweb/bioweb/cache/")
                print("解压完成")
            bio_ClassAndFunction.cal_thick(99)    # 调用处理函数
            print("处理完成")


        return HttpResponseRedirect(reverse('mainsite:success'))

    return render(request, 'mainsite/cal_thick_compress.html')

@login_required
def success(request):
    # 上传成功的界面
    return render(request, 'mainsite/success.html')

@login_required
def success_download_excel(request):
    # 下载excel文件的响应
    filename = "/home/zyf/PycharmProjects/bioweb/bioweb/cache/result.xls"
    the_file_name = 'result.xls'
    file = open(filename, 'rb')
    response = FileResponse(file)
    response['Content-Type'] = 'application/octet-stream'
    response['Content-Disposition'] = 'attachment;filename="{0}"'.format(the_file_name)
    return response

# Create your views here.
