from django.urls import path,include

from . import views

app_name = 'mainsite'

urlpatterns = [
    # 主页
    path('', views.index, name='index'),

    # 显示两中计算方式的入口界面
    path('cal/', views.calculation, name='calculation'),

    # 上传图片并计算-中轴线、宽度、半径计算界面
    path('cal/1', views.cal_dia_centerline_width, name='cal1'),

    # 上传压缩包计算中轴线、宽度、半径计算界面
    path('cal/1/compress', views.cal_dia_centerline_width_compress_fake, name='cal1compress'),
    path('cal/1/dia_compress_real', views.cal_dia_centerline_width_compress, name='cal1compress_real'),


    # 上传图片并计算厚度的界面
    path('cal/2', views.cal_thickness, name='cal2'),
    # 上传压缩包计算厚度的界面
    path('cal/2/compress', views.cal_thickness_compress_fake, name='cal2compress'),
    path('cal/2/thick_compress_real', views.cal_thickness_compress, name='cal2compress_real'),

    # 下载url
    path('cal/download', views.success_download_excel, name='download'),

    # 上传成功界面
    path('cal/success', views.success, name='success'),


    path('show_uploade_progress', views.show_uploade_progress),

    path('upload_compress_1', views.up_load_compressfile_1, name='upload_compress_1'),
    path('upload_compress_2', views.up_load_compressfile_2, name='upload_compress_2'),
    path('show_progress', views.show_progress),

]
