import urllib
from django.shortcuts import render
from pages.models import Customer
import numpy as np
from .apps import *
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, JSONParser
# import cloudinary.uploader
# import cloudinary
from django.core.files.base import ContentFile
#import matplotlib.pyplot as plt
from django.http import HttpResponse
import cv2
#from .serializers import ImageSerializer
####swap
import sys
import getopt
#import cv2 #import numpy
from components.delaunay_triangulation import find_delauney_triangulation
from components.affine_transformation import apply_affine_transformation
from components.clone_mask import merge_mask_with_image
from components.landmark_detection import detect_landmarks
from components.convex_hull import find_convex_hull
from .serializers import ImageSerializer
####
# from django.conf import settings
from .throttling import CustomAnonRateThrottle, CustomUserRateThrottle
# cloudinary = settings.CLOUDINARY_STORAGE
# import cloudinary.uploader

def home(request):
    return HttpResponse('<h2>Welcome to Face-Swap Home</h2>')

# Create your views here.

################Firebase###############
from django.conf import settings
from django.core.files.storage import default_storage
from django.contrib import messages
import pyrebase
import os
from decouple import config

configFire = {
    "apiKey": config('apiKey'),
    "authDomain": config('authDomain'),
    "projectId": config('projectId'),
    "storageBucket": config('storageBucket'),
    "messagingSenderId": config('messagingSenderId'),
    "appId": config('appId'),
    "measurementId": config('measurementId'),
    "databaseURL": "",
}

firebase = pyrebase.initialize_app(configFire) 
storage = firebase.storage()
############################


class SwapUploadView(APIView):
    parser_classes = (
        MultiPartParser,
        JSONParser,
    )

    @staticmethod
    def post(request):
        file1 = request.data.get('style')
        file2 = request.data.get('pic')
        #cloudinary
        #upload_data1 = cloudinary.uploader.upload(file1)
        #upload_data2 = cloudinary.uploader.upload(file2)#, public_id = "iotneu")
        #firebase
        upload_data1 = default_storage.save('style', file1)
        storage.child("files/" + 'style').put("media/" + 'style')
        upload_data2 = default_storage.save('pic', file2)
        storage.child("files/" + 'picture').put("media/" + 'picture')
        ####
        first_image = upload_data1['url']
        second_image = upload_data2['url']
        if not first_image or not second_image:
            return Response({
                'status': 'failed',
                'data': 'incorect - parameters',
                }, status=400)
###########################################
        images = Customer(style=first_image, pic=second_image)
        images.save()  
        #final new#offline
        # first_image = str(images.style)
        # second_image = str(images.pic)
        # img1 = "." + "/media/" + first_image
        # img2 = "." + "/media/" + second_image
        #################
        # online
        img1 = urllib.request.urlopen(str(images.style))
        img2 = urllib.request.urlopen(str(images.pic))
        list_of_images = [img1, img2]
        #################
        argv=None
        try:
            opts, args = getopt.getopt(argv, "hi:", ["ifile="])
        except getopt.GetoptError:
            return Response({
                'status': 'failed',
                'data': 'null',
                }, status=400)

        for opt, arg in opts:
            if opt in ("-i", "--ifile"):
                list_of_images.append(arg)
            else:
                return Response({
                'status': 'failed',
                'data': 'null',
                }, status=400)
    
        arrIMG1 = np.asarray(bytearray(img1.read()), dtype=np.uint8)
        arrIMG2 = np.asarray(bytearray(img2.read()), dtype=np.uint8)
        image_1 = cv2.imdecode(arrIMG1, -1)
        image_2 = cv2.imdecode(arrIMG2, -1)
        #req = urllib.urlopen('http://answers.opencv.org/upfiles/logo_2.png')
        #img = cv2.imdecode(arr, -1)

        #print (image_1)[0]
        landmarks_of_image1 = detect_landmarks(image_1)[0]
        landmarks_of_image2 = detect_landmarks(image_2)[0]
        convex_image1, convex_image2 = find_convex_hull(landmarks_of_image1, landmarks_of_image2, image_1, image_2)
        triangulation_image1 = find_delauney_triangulation(image_1, convex_image1)
        triangulation_image2 = find_delauney_triangulation(image_2, convex_image2)
        user_to_hairstyle1 = apply_affine_transformation(triangulation_image1, convex_image1, convex_image2, image_1, image_2)
        user_to_hairstyle2 = apply_affine_transformation(triangulation_image2, convex_image2, convex_image1, image_2, image_1)
        output_hairstyle_user = merge_mask_with_image(convex_image2, user_to_hairstyle1, image_2)
        output_user_hairstyle = merge_mask_with_image(convex_image1, user_to_hairstyle2, image_1)
        print ('level 1')
        ret, buf = cv2.imencode('.jpg', user_to_hairstyle2) # cropped_image: cv2 / np array
        content = ContentFile(buf.tobytes())
        print ('level 2')
        img_model = images
        img_model.merge.save('output.jpg', content)
        print ('level 3')
        #######
        return Response({
            'status': 'success',
            'output': img_model.merge,
            'resnetCT_chest_pred':'success',
        }, status=201)
############################################

        #load models
        #resnet_chest = load_model('./ml/models/resnet_ct.h5')
        req = urllib.request.urlopen(img)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        image = cv2.imdecode(arr, -1) # 'Load it as it is'
        #image = cv2.imread('upload_chest.jpg') # read file 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # arrange format as per keras
        image = cv2.resize(image,(224,224))
        image = np.array(image) / 255
        image = np.expand_dims(image, axis=0)
        
        #print("Resnet Predictions:")
        if probability[0] > 0.5:
            resnet_chest_pred = str('%.2f' % (probability[0]*100) + '% COVID') 
        else:
            resnet_chest_pred = str('%.2f' % ((1-probability[0])*100) + '% NonCOVID')
        return Response({
            'status': 'success',
            'data': upload_data,
            #'xceptionCT_chest_pred':xception_chest_pred,
            #'inceptionCT_chest_pred':inception_chest_pred,
            #'vggCT_chest_pred':vgg_chest_pred,
            'resnetCT_chest_pred':resnet_chest_pred,
        }, status=201)


class SwapUploadSerView(APIView):
    parser_classes = (
        MultiPartParser,
        JSONParser,
    )

    @staticmethod
    def post(request):
        serializer = ImageSerializer(data=request.data)
        if serializer.is_valid():
            file1 = serializer.validated_data['style'] #getting a selected field from a serializer 
            file2 = serializer.validated_data['pic'] #getting a selected field from a serializer 
            # upload_data = cloudinary.uploader.upload(file1)
            upload_data1 = default_storage.save('style', file1) #save2localcomp
            upload_dataA = "." + "/media/" + upload_data1
            cloud1 = storage.child("files/faceswap_images/" + str(upload_data1)).put(upload_dataA)#upload
            first_image = storage.child("files/faceswap_images/" + str(upload_data1)).get_url(None)#get urlforPic1
            upload_data2 = default_storage.save('pic', file2)
            upload_dataB = "." + "/media/" + upload_data2
            cloud2 = storage.child("files/faceswap_images/"+ str(upload_data2)).put(upload_dataB)
            second_image = storage.child("files/faceswap_images/" + str(upload_data2)).get_url(None)
        else:
            return Response({
                'status': 'failed',
                'data': 'incorect - parameters',
                }, status=400)
        # if not first_image or not second_image:
        ###########################################
        img1 = urllib.request.urlopen(str(first_image))
        img2 = urllib.request.urlopen(str(second_image))
        list_of_images = [img1, img2]
        #################
        argv=None
        try:
            opts, args = getopt.getopt(argv, "hi:", ["ifile="])
        except getopt.GetoptError:
            return Response({
                'status': 'failed',
                'data': 'null',
                }, status=400)

        for opt, arg in opts:
            if opt in ("-i", "--ifile"):
                list_of_images.append(arg)
            else:
                return Response({
                'status': 'failed',
                'data': 'null',
                }, status=400)
    
        arrIMG1 = np.asarray(bytearray(img1.read()), dtype=np.uint8)#readImageasArray
        arrIMG2 = np.asarray(bytearray(img2.read()), dtype=np.uint8)
        image_1 = cv2.imdecode(arrIMG1, -1)#return it as it is
        image_2 = cv2.imdecode(arrIMG2, -1)
        #print (image_1)[0]
        landmarks_of_image1 = detect_landmarks(image_1)[0]
        landmarks_of_image2 = detect_landmarks(image_2)[0]
        convex_image1, convex_image2 = find_convex_hull(landmarks_of_image1, landmarks_of_image2, image_1, image_2)
        triangulation_image1 = find_delauney_triangulation(image_1, convex_image1)
        triangulation_image2 = find_delauney_triangulation(image_2, convex_image2)
        user_to_hairstyle1 = apply_affine_transformation(triangulation_image1, convex_image1, convex_image2, image_1, image_2)
        user_to_hairstyle2 = apply_affine_transformation(triangulation_image2, convex_image2, convex_image1, image_2, image_1)
        output_hairstyle_user = merge_mask_with_image(convex_image2, user_to_hairstyle1, image_2)
        output_user_hairstyle = merge_mask_with_image(convex_image1, user_to_hairstyle2, image_1)
        print ('level 1')
        ret, buf = cv2.imencode('.jpg', user_to_hairstyle2) # cropped_image: cv2 / np array
        content = ContentFile(buf.tobytes())
        print ('level 2')
        img_model = Customer(style=first_image, pic=second_image)
        img_model.save()  
        print ('level 3')
        #################final final------------
        upload_data3 = default_storage.save('output.jpg', content)
        upload_dataC = "." + "/media/" + upload_data3
        cloud3 = storage.child("files/faceswap_images/"+ str(upload_data3)).put(upload_dataC)
        third_image = storage.child("files/faceswap_images/" + str(upload_data3)).get_url(None)
        img_model.merge = third_image
        img_model.save() #img_model.merge.save('output.jpg', third_image)
        #######
        return Response({
            'status': 'success',
            'style': str(img_model.style),
            'picture': str(img_model.pic),
            'output': str(img_model.merge),
        }, status=201)
############################################

        
class SwapUploadViewLimit(APIView):
    throttle_classes = (CustomAnonRateThrottle, )#view based throtting - overiding the settings.py 
    parser_classes = (
        MultiPartParser,
        JSONParser,
    )

    @staticmethod
    def post(request):
        file1 = request.data.get('style')
        file2 = request.data.get('pic')
        #cloudinary
        if file1:
            print ('file1 received')
        if file2:
            print ('file2 received')
        if file1 and file2:
            upload_data1 = default_storage.save('style', file1) #save2localcomp
            upload_dataA = "." + "/media/" + upload_data1
            cloud1 = storage.child("files/faceswap_images/" + str(upload_data1)).put(upload_dataA)#upload
            first_image = storage.child("files/faceswap_images/" + str(upload_data1)).get_url(None)#get urlforPic1
            upload_data2 = default_storage.save('pic', file2)
            upload_dataB = "." + "/media/" + upload_data2
            cloud2 = storage.child("files/faceswap_images/"+ str(upload_data2)).put(upload_dataB)
            second_image = storage.child("files/faceswap_images/" + str(upload_data2)).get_url(None)
        else:
            return Response({
                'status': 'failed',
                'data': 'incorect - parameters',
                }, status=400)
        ###########################################
        img1 = urllib.request.urlopen(str(first_image))
        img2 = urllib.request.urlopen(str(second_image))
        list_of_images = [img1, img2]
        #################
        argv=None
        try:
            opts, args = getopt.getopt(argv, "hi:", ["ifile="])
        except getopt.GetoptError:
            return Response({
                'status': 'failed',
                'data': 'null',
                }, status=400)

        for opt, arg in opts:
            if opt in ("-i", "--ifile"):
                list_of_images.append(arg)
            else:
                return Response({
                'status': 'failed',
                'data': 'null',
                }, status=400)

        arrIMG1 = np.asarray(bytearray(img1.read()), dtype=np.uint8)#readImageasArray
        arrIMG2 = np.asarray(bytearray(img2.read()), dtype=np.uint8)
        image_1 = cv2.imdecode(arrIMG1, -1)#return it as it is
        image_2 = cv2.imdecode(arrIMG2, -1)
        #print (image_1)[0]
        landmarks_of_image1 = detect_landmarks(image_1)[0]
        landmarks_of_image2 = detect_landmarks(image_2)[0]
        convex_image1, convex_image2 = find_convex_hull(landmarks_of_image1, landmarks_of_image2, image_1, image_2)
        triangulation_image1 = find_delauney_triangulation(image_1, convex_image1)
        triangulation_image2 = find_delauney_triangulation(image_2, convex_image2)
        user_to_hairstyle1 = apply_affine_transformation(triangulation_image1, convex_image1, convex_image2, image_1, image_2)
        user_to_hairstyle2 = apply_affine_transformation(triangulation_image2, convex_image2, convex_image1, image_2, image_1)
        output_hairstyle_user = merge_mask_with_image(convex_image2, user_to_hairstyle1, image_2)
        output_user_hairstyle = merge_mask_with_image(convex_image1, user_to_hairstyle2, image_1)
        print ('level 1')
        ret, buf = cv2.imencode('.jpg', user_to_hairstyle2) # cropped_image: cv2 / np array
        content = ContentFile(buf.tobytes())
        print ('level 2')
        img_model = Customer(style=first_image, pic=second_image)
        img_model.save()  
        print ('level 3')
        #################final final------------
        upload_data3 = default_storage.save('output.jpg', content)
        upload_dataC = "." + "/media/" + upload_data3
        cloud3 = storage.child("files/faceswap_images/"+ str(upload_data3)).put(upload_dataC)
        third_image = storage.child("files/faceswap_images/" + str(upload_data3)).get_url(None)
        img_model.merge = third_image
        img_model.save() #img_model.merge.save('output.jpg', third_image)
        #######
        return Response({
            'status': 'success',
            'style': str(img_model.style),
            'picture': str(img_model.pic),
            'output': str(img_model.merge),
        }, status=201)
############################################