from abc import ABC
from django.shortcuts import render, redirect
from django.contrib import messages
import os
#MLO 
import sys
import getopt
import cv2
import numpy
from components.delaunay_triangulation import find_delauney_triangulation
from components.affine_transformation import apply_affine_transformation
from components.clone_mask import merge_mask_with_image
from components.landmark_detection import detect_landmarks
from components.convex_hull import find_convex_hull
import io
from PIL import Image
from .models import Customer, Style
from django.core.files.base import ContentFile


# Create your views here.
def index(request):
    my_dict = {"insert_me": "I am from views.py"}
    return render(request,'pages/index.html',context=my_dict)



def upload_new(request, argv=None):
    expected_files = 2
    image = Style.objects.values('style', 'id', 'name')
    
    # key = image.id
    my_dict = {"output": image,
              "key": "I am from views.py"}
            #"output": ['1', '2', '3'],

    if request.method == 'POST':
        input1 = request.POST['DropdownListName']
        if input1:
            # itemValue = input1['your_filed_name'].value()
            print(input1)
        input2 = request.FILES['image1']#.read()
        if input2:
            print('Received2')
        if not (input1 or input2):
            messages.error(request,"Missing Picture or Style")
            return redirect('pages-index')
        #MLO
        in_image= Style.objects.get(id=input1)
        print(in_image.style)
        first_image = str(in_image.style)
        # print(in_image.uuid)
        # print(in_image.id)
        # print(in_image.style)
        upload_data2 = default_storage.save('pic', input2)
        upload_dataB = "." + "/media/" + upload_data2
        cloud2 = storage.child("files/faceswap_images/"+ str(upload_data2)).put(upload_dataB)
        second_image = storage.child("files/faceswap_images/" + str(upload_data2)).get_url(None)

        ###########################################
        img1 = urllib.request.urlopen(str(in_image.style))
        img2 = urllib.request.urlopen(str(second_image))
        list_of_images = [img1, img2]
        #################
        try:
            opts, args = getopt.getopt(argv, "hi:", ["ifile="])
        except getopt.GetoptError:
            messages.error(request,"GetoptError")#exit_error()
            return redirect('pages-index')

        for opt, arg in opts:
            if opt in ("-i", "--ifile"):
                list_of_images.append(arg)
            else:
                messages.error(request,"Exit error")
                return redirect('pages-index')

    
        if len(list_of_images) != expected_files:
            messages.error(request,"Exit error")
            return redirect('pages-index')

        #decode image
        arrIMG1 = np.asarray(bytearray(img1.read()), dtype=np.uint8)#readImageasArray
        arrIMG2 = np.asarray(bytearray(img2.read()), dtype=np.uint8)
        image_1 = cv2.imdecode(arrIMG1, -1)#return it as it is
        image_2 = cv2.imdecode(arrIMG2, -1)
        #end decode image f
        landmarks_of_image1 = detect_landmarks(image_1)[0]
        landmarks_of_image2 = detect_landmarks(image_2)[0]
        convex_image1, convex_image2 = find_convex_hull(landmarks_of_image1, landmarks_of_image2, image_1, image_2)
        triangulation_image1 = find_delauney_triangulation(image_1, convex_image1)
        triangulation_image2 = find_delauney_triangulation(image_2, convex_image2)
        user_to_hairstyle1 = apply_affine_transformation(triangulation_image1, convex_image1, convex_image2, image_1, image_2)
        user_to_hairstyle2 = apply_affine_transformation(triangulation_image2, convex_image2, convex_image1, image_2, image_1)
        output_hairstyle_user = merge_mask_with_image(convex_image2, user_to_hairstyle1, image_2)
        output_user_hairstyle = merge_mask_with_image(convex_image1, user_to_hairstyle2, image_1)
        ##### GDAL
        #save to django model
        ret, buf = cv2.imencode('.jpg', user_to_hairstyle2) # cropped_image: cv2 / np array
        content = ContentFile(buf.tobytes())
        print ('level 2')
        img_model =  Customer()#images
        # img_model.save()  
        print ('level 3')
        #################final final------------
        upload_data3 = default_storage.save('output.jpg', content)
        upload_dataC = "." + "/media/" + upload_data3
        cloud3 = storage.child("files/faceswap_images/"+ str(upload_data3)).put(upload_dataC)
        third_image = storage.child("files/faceswap_images/" + str(upload_data3)).get_url(None)
        img_model.style = first_image
        img_model.pic = second_image
        img_model.merge = third_image
        img_model.save() #img_model.merge.save('output.jpg', third_image)
        #######
        my_dict = {
            "name": 'user-name',
            "output": img_model,
        }
        return render(request,'pages/outputurl.html',context=my_dict)        
    #MLO end
    return render(request,'pages/upload2.html',context=my_dict)     

###################

def upload(request, argv=None):
    expected_files = 2
    my_dict = {"insert_me": "I am from views.py"}

    if request.method == 'POST':
        input1 = request.FILES['image1']
        if input1:
            print('Received1')
        input2 = request.FILES['image2']#.read()
        if input2:
            print('Received2')
        if not input1 or input2:
            pass# return redirect('appointment')
        #MLO
        images = Customer(style=input1, pic=input2)
        
        images.save()  

        list_of_images = [images.style, images.pic]
        #final new
        first_image = str(images.style)
        second_image = str(images.pic)
        img1 = "." + "/media/" + first_image
        img2 = "." + "/media/" + second_image
        try:
            opts, args = getopt.getopt(argv, "hi:", ["ifile="])
        except getopt.GetoptError:
            messages.error(request,"GetoptError")#exit_error()
            return redirect('pages-index')

        for opt, arg in opts:
            if opt in ("-i", "--ifile"):
                list_of_images.append(arg)
            else:
                messages.error(request,"Exit error")
                return redirect('pages-index')

    
        if len(list_of_images) != expected_files:
            messages.error(request,"Exit error")
            return redirect('pages-index')
        image_1 = img1
        image_2 = img2
        image_1 = cv2.imread(str(img1)) #modify here # cannot convert to str for filename
        image_2 = cv2.imread(str(img2)) #modify here # cannot convert to str for filename

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
        
        ##### GDAL
        #save to django model
        print ('level 1')
        # ret, buf = cv2.imencode('.jpg', output_user_hairstyle) # cropped_image: cv2 / np array
        ret, buf = cv2.imencode('.jpg', user_to_hairstyle2) # cropped_image: cv2 / np array
        content = ContentFile(buf.tobytes())
        print ('level 2')
        img_model = images
        img_model.merge.save('output.jpg', content)
        #######
        my_dict = {
            "name": 'user-name',
            "output": img_model,
        }
        return render(request,'pages/output.html',context=my_dict)        

        
    #MLO end
    return render(request,'pages/upload.html',context=my_dict)        


def check(request):
    img_model = Customer.objects.filter(merge__isnull=False).exclude(merge__exact='')#ex blanc d non
    my_dict = {
        "name": 'user-name',
        "output": img_model,
    }
    return render(request,'pages/output2.html',context=my_dict) 



###################
import imageio
from faceswap import FaceSwap, NoFaceException, MoreThanOneFaceException

def upload_fl(request):
    expected_files = 2
    my_dict = {"insert_me": "I am from views.py"}

    if request.method == 'POST':
        input1 = request.FILES['image1']#.read()#.read() for Type-error-Can't convert object of type 'InMemoryUploadedFile' to 'str' for 'filename'
        if input1:
            print('Received1')
        input2 = request.FILES['image2']#.read()
        if input2:
            print('Received2')
        if not input1 or input2:
            pass# return redirect('appointment')
        #MLO
        images = Customer(style=input1, pic=input2)
        
        images.save()  

        list_of_images = [images.style, images.pic]
        print(str(images.style))
        print(str(images.pic))
        first_image = str(images.style)
        second_image = str(images.pic)
        img1 = "." + "/media/" + first_image
        img2 = "." + "/media/" + second_image
        print (img1)
    ######################
		#Merge Faces
        try:
            FaceSwap(img1, img2)
        except NoFaceException:
            print("A Face could not be detected!")
			# flash("A Face could not be detected! Only human faces are detected", "danger")
			# return render_template('main.html', form=form)
        except MoreThanOneFaceException:#
            print("There seems to be more than one face in the image")

	#Give Back Image
        img_model = images
        #save to django model
        print ('level 1')
        ret, buf = cv2.imencode('.jpg', output_im) # cropped_image: cv2 / np array
        content = ContentFile(buf.tobytes())
        print ('level 2')
        img_model.merge.save('output.jpg', content)
        # img_model = Customer()
        #######
        my_dict = {
            "name": 'user-name',
            "output": img_model,
        }
        return render(request,'pages/output.html',context=my_dict)        
    return render(request,'pages/upload.html',context=my_dict)        


##########################
def get_index(arr):
        index = 0
        if arr[0]:
            index = arr[0][0]
        return index

def get_landmarks(landmarks, landmarks_points):
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x, y))


import base64
from base64 import encodebytes
import numpy as np
import io
import dlib

   
################Firebase###############
from django.conf import settings
from django.core.files.storage import default_storage
#from django.contrib import messages
import pyrebase
import os
import urllib
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

#############CLOUD-UPLOAD#################
def upload_cloud(request, argv=None):
    expected_files = 2
    my_dict = {"insert_me": "I am from views.py"}

    if request.method == 'POST':
        input1 = request.FILES['image1']#.read()#.read() for Type-error-Can't convert object of type 'InMemoryUploadedFile' to 'str' for 'filename'
        if input1:
            print('Received1')
        input2 = request.FILES['image2']#.read()
        if input2:
            print('Received2')
        if not input1 or input2:
            pass# return redirect('appointment')
        #MLO
        # images = Customer(style=input1, pic=input2)##############
        
        # images.save()  

        # list_of_images = [images.style, images.pic]############
        #final new
        # first_image = str(images.style)
        # second_image = str(images.pic)
        upload_data1 = default_storage.save('style', input1) #save2localcomp
        upload_dataA = "." + "/media/" + upload_data1
        cloud1 = storage.child("files/faceswap_images/" + str(upload_data1)).put(upload_dataA)#upload
        first_image = storage.child("files/faceswap_images/" + str(upload_data1)).get_url(None)#get urlforPic1
        upload_data2 = default_storage.save('pic', input2)
        upload_dataB = "." + "/media/" + upload_data2
        cloud2 = storage.child("files/faceswap_images/"+ str(upload_data2)).put(upload_dataB)
        second_image = storage.child("files/faceswap_images/" + str(upload_data2)).get_url(None)

        ###########################################
        img1 = urllib.request.urlopen(str(first_image))
        img2 = urllib.request.urlopen(str(second_image))
        list_of_images = [img1, img2]
        #################
        try:
            opts, args = getopt.getopt(argv, "hi:", ["ifile="])
        except getopt.GetoptError:
            messages.error(request,"GetoptError")#exit_error()
            return redirect('pages-index')

        for opt, arg in opts:
            if opt in ("-i", "--ifile"):
                list_of_images.append(arg)
            else:
                messages.error(request,"Exit error")
                return redirect('pages-index')

    
        if len(list_of_images) != expected_files:
            messages.error(request,"Exit error")
            return redirect('pages-index')

        #decode image
        arrIMG1 = np.asarray(bytearray(img1.read()), dtype=np.uint8)#readImageasArray
        arrIMG2 = np.asarray(bytearray(img2.read()), dtype=np.uint8)
        image_1 = cv2.imdecode(arrIMG1, -1)#return it as it is
        image_2 = cv2.imdecode(arrIMG2, -1)
        #end decode image f
        landmarks_of_image1 = detect_landmarks(image_1)[0]
        landmarks_of_image2 = detect_landmarks(image_2)[0]
        convex_image1, convex_image2 = find_convex_hull(landmarks_of_image1, landmarks_of_image2, image_1, image_2)
        triangulation_image1 = find_delauney_triangulation(image_1, convex_image1)
        triangulation_image2 = find_delauney_triangulation(image_2, convex_image2)
        user_to_hairstyle1 = apply_affine_transformation(triangulation_image1, convex_image1, convex_image2, image_1, image_2)
        user_to_hairstyle2 = apply_affine_transformation(triangulation_image2, convex_image2, convex_image1, image_2, image_1)
        output_hairstyle_user = merge_mask_with_image(convex_image2, user_to_hairstyle1, image_2)
        output_user_hairstyle = merge_mask_with_image(convex_image1, user_to_hairstyle2, image_1)
        ##### GDAL
        #save to django model
        ret, buf = cv2.imencode('.jpg', user_to_hairstyle2) # cropped_image: cv2 / np array
        content = ContentFile(buf.tobytes())
        print ('level 2')
        img_model =  Customer()#images
        # img_model.save()  
        print ('level 3')
        #################final final------------
        upload_data3 = default_storage.save('output.jpg', content)
        upload_dataC = "." + "/media/" + upload_data3
        cloud3 = storage.child("files/faceswap_images/"+ str(upload_data3)).put(upload_dataC)
        third_image = storage.child("files/faceswap_images/" + str(upload_data3)).get_url(None)
        img_model.style = first_image
        img_model.pic = second_image
        img_model.merge = third_image
        img_model.save() #img_model.merge.save('output.jpg', third_image)
        #######
        my_dict = {
            "name": 'user-name',
            "output": img_model,
        }
        return render(request,'pages/outputurl.html',context=my_dict)        

    #MLO end
    return render(request,'pages/upload.html',context=my_dict)        


from django.views import generic
from . import forms

class StyleListView(generic.ListView):
    # model = Style
    form_class = forms.StyleForm
    queryset = Style.objects.all().order_by('-id')
    context_object_name = 'style'
    paginate_by = 7

def StyleCreateView(request):
    expected_files = 2
    my_dict = {"insert_me": "I am from views.py"}

    if request.method == 'POST':
        stylename = request.POST['stylename']
        input1 = request.FILES['image1']#.read()#.read() for Type-error-Can't convert object of type 'InMemoryUploadedFile' to 'str' for 'filename'
        print (stylename, input1.name )
        if input1:
            print('Received1')
        if not input1:
            messages.error(request,"Style Image not supplied")#exit_error()
            return redirect('pages-index')
        first_image = default_storage.save('style', input1) #save2localcomp
        upload_dataA = "." + "/media/" + first_image
        cloud1 = storage.child("files/faceswap_images/" + str(first_image)).put(upload_dataA)#upload
        first_image = storage.child("files/faceswap_images/" + str(first_image)).get_url(None)#get urlforPic1
        
        style_model =  Style(name=stylename, style=first_image)#images
        style_model.save() #img_model.m
        #######
        my_dict = {
            # "name": 'user-name',
            "output": style_model,
        }
        return render(request,'pages/outputstyle.html',context=my_dict)        
    return render(request,'pages/style_create_form.html')        


class StyleDetailView(generic.DetailView):
    model = Style
    form_class = forms.StyleForm
    template_name = 'pages/style_detail.html'   
    def get_object(self, queryset=None):
        return Style.objects.get(uuid=self.kwargs.get("uuid"))
    
    # def get_object(self):
    #     object = get_object_or_404(Style,title=self.kwargs['title'])
    #     return object

class StyleUpdateView(generic.UpdateView):
    # pk_url_kwarg = "uuid"  
    model = Style
    form_class = forms.StyleForm
    def get_object(self, queryset=None): #rejects # pk_url_kwarg = "uuid"  
        return Style.objects.get(uuid=self.kwargs.get("uuid"))

    def form_valid(self, form):
        input1 = self.request.FILES['style']# print(input1.name)
        #uploadCloudStart
        first_image = default_storage.save('style', input1) #save2localcomp
        upload_dataA = "." + "/media/" + first_image
        cloud1 = storage.child("files/faceswap_images/" + str(first_image)).put(upload_dataA)#upload
        first_image = storage.child("files/faceswap_images/" + str(first_image)).get_url(None)#get urlforPic1
        form.instance.style = first_image
        return super().form_valid(form)        

#################################
#extra############EXTRAAAAAAAAAAAAAAAAAAAAAAAAA

def upload_cloud_new(request, argv=None):
    expected_files = 2
    my_dict = {"insert_me": "I am from views.py"}

    if request.method == 'POST':
        input1 = request.FILES['image1']#.read()#.read() for Type-error-Can't convert object of type 'InMemoryUploadedFile' to 'str' for 'filename'
        if input1:
            print('Received1')
        input2 = request.FILES['image2']#.read()
        if input2:
            print('Received2')
        if not input1 or input2:
            pass# return redirect('appointment')
        #MLO
        upload_data1 = default_storage.save('style', input1) #save2localcomp
        upload_dataA = "." + "/media/" + upload_data1
        cloud1 = storage.child("files/faceswap_images/" + str(upload_data1)).put(upload_dataA)#upload
        first_image = storage.child("files/faceswap_images/" + str(upload_data1)).get_url(None)#get urlforPic1
        upload_data2 = default_storage.save('pic', input2)
        upload_dataB = "." + "/media/" + upload_data2
        cloud2 = storage.child("files/faceswap_images/"+ str(upload_data2)).put(upload_dataB)
        second_image = storage.child("files/faceswap_images/" + str(upload_data2)).get_url(None)

        ###########################################
        img1 = urllib.request.urlopen(str(first_image))
        img2 = urllib.request.urlopen(str(second_image))
        list_of_images = [img1, img2]
        #################
        try:
            opts, args = getopt.getopt(argv, "hi:", ["ifile="])
        except getopt.GetoptError:
            messages.error(request,"GetoptError")#exit_error()
            return redirect('pages-index')

        for opt, arg in opts:
            if opt in ("-i", "--ifile"):
                list_of_images.append(arg)
            else:
                messages.error(request,"Exit error")
                return redirect('pages-index')

    
        if len(list_of_images) != expected_files:
            messages.error(request,"Exit error")
            return redirect('pages-index')

        #decode image
        arrIMG1 = np.asarray(bytearray(img1.read()), dtype=np.uint8)#readImageasArray
        arrIMG2 = np.asarray(bytearray(img2.read()), dtype=np.uint8)
        image_1 = cv2.imdecode(arrIMG1, -1)#return it as it is
        image_2 = cv2.imdecode(arrIMG2, -1)
        #end decode image f
        landmarks_of_image1 = detect_landmarks(image_1)[0]
        landmarks_of_image2 = detect_landmarks(image_2)[0]
        convex_image1, convex_image2 = find_convex_hull(landmarks_of_image1, landmarks_of_image2, image_1, image_2)
        triangulation_image1 = find_delauney_triangulation(image_1, convex_image1)
        triangulation_image2 = find_delauney_triangulation(image_2, convex_image2)
        user_to_hairstyle1 = apply_affine_transformation(triangulation_image1, convex_image1, convex_image2, image_1, image_2)
        user_to_hairstyle2 = apply_affine_transformation(triangulation_image2, convex_image2, convex_image1, image_2, image_1)
        output_hairstyle_user = merge_mask_with_image(convex_image2, user_to_hairstyle1, image_2)
        output_user_hairstyle = merge_mask_with_image(convex_image1, user_to_hairstyle2, image_1)
        ##### GDAL
        #save to django model
        ret, buf = cv2.imencode('.jpg', user_to_hairstyle2) # cropped_image: cv2 / np array
        content = ContentFile(buf.tobytes())
        print ('level 2')
        img_model =  Customer()#images
        # img_model.save()  
        print ('level 3')
        #################final final------------
        upload_data3 = default_storage.save('output.jpg', content)
        upload_dataC = "." + "/media/" + upload_data3
        cloud3 = storage.child("files/faceswap_images/"+ str(upload_data3)).put(upload_dataC)
        third_image = storage.child("files/faceswap_images/" + str(upload_data3)).get_url(None)
        img_model.style = first_image
        img_model.pic = second_image
        img_model.merge = third_image
        img_model.save() #img_model.merge.save('output.jpg', third_image)
        #######
        my_dict = {
            "name": 'user-name',
            "output": img_model,
        }
        return render(request,'pages/outputurl.html',context=my_dict)        

    #MLO end
    return render(request,'pages/upload.html',context=my_dict)        
