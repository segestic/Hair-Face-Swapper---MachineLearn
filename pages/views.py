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
#MLO end

# Create your views here.
def index(request):
    my_dict = {"insert_me": "I am from views.py"}
    return render(request,'pages/index.html',context=my_dict)


# def upload(request):
#     my_dict = {"insert_me": "I am from views"}
#     return render(request,'pages/upload.html',context=my_dict)

# if 'message_frm' in request.POST:
########
import io
# from PIL import Image

# im = Image.open('test.jpg')
# im_resize = im.resize((500, 500))
# buf = io.BytesIO()
# im_resize.save(buf, format='JPEG')
# byte_im = buf.getvalue()
#######
from PIL import Image
from .models import Customer, Style
from django.core.files.base import ContentFile



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
        print(in_image)
        first_image = str(in_image)
        print(in_image.uuid)
        print(in_image.id)
        print(in_image.style)
        # upload_data2 = default_storage.save('pic', input2)
        # upload_dataB = "." + "/media/" + upload_data2
        # cloud2 = storage.child("files/faceswap_images/"+ str(upload_data2)).put(upload_dataB)
        # second_image = storage.child("files/faceswap_images/" + str(upload_data2)).get_url(None)

        # ###########################################
        # img1 = urllib.request.urlopen(str(in_image))
        # img2 = urllib.request.urlopen(str(second_image))
        # list_of_images = [img1, img2]
        # #################
        # try:
        #     opts, args = getopt.getopt(argv, "hi:", ["ifile="])
        # except getopt.GetoptError:
        #     messages.error(request,"GetoptError")#exit_error()
        #     return redirect('pages-index')

        # for opt, arg in opts:
        #     if opt in ("-i", "--ifile"):
        #         list_of_images.append(arg)
        #     else:
        #         messages.error(request,"Exit error")
        #         return redirect('pages-index')

    
        # if len(list_of_images) != expected_files:
        #     messages.error(request,"Exit error")
        #     return redirect('pages-index')

        # #decode image
        # arrIMG1 = np.asarray(bytearray(img1.read()), dtype=np.uint8)#readImageasArray
        # arrIMG2 = np.asarray(bytearray(img2.read()), dtype=np.uint8)
        # image_1 = cv2.imdecode(arrIMG1, -1)#return it as it is
        # image_2 = cv2.imdecode(arrIMG2, -1)
        # #end decode image f
        # landmarks_of_image1 = detect_landmarks(image_1)[0]
        # landmarks_of_image2 = detect_landmarks(image_2)[0]
        # convex_image1, convex_image2 = find_convex_hull(landmarks_of_image1, landmarks_of_image2, image_1, image_2)
        # triangulation_image1 = find_delauney_triangulation(image_1, convex_image1)
        # triangulation_image2 = find_delauney_triangulation(image_2, convex_image2)
        # user_to_hairstyle1 = apply_affine_transformation(triangulation_image1, convex_image1, convex_image2, image_1, image_2)
        # user_to_hairstyle2 = apply_affine_transformation(triangulation_image2, convex_image2, convex_image1, image_2, image_1)
        # output_hairstyle_user = merge_mask_with_image(convex_image2, user_to_hairstyle1, image_2)
        # output_user_hairstyle = merge_mask_with_image(convex_image1, user_to_hairstyle2, image_1)
        # ##### GDAL
        # #save to django model
        # ret, buf = cv2.imencode('.jpg', user_to_hairstyle2) # cropped_image: cv2 / np array
        # content = ContentFile(buf.tobytes())
        # print ('level 2')
        # img_model =  Customer()#images
        # # img_model.save()  
        # print ('level 3')
        # #################final final------------
        # upload_data3 = default_storage.save('output.jpg', content)
        # upload_dataC = "." + "/media/" + upload_data3
        # cloud3 = storage.child("files/faceswap_images/"+ str(upload_data3)).put(upload_dataC)
        # third_image = storage.child("files/faceswap_images/" + str(upload_data3)).get_url(None)
        # img_model.style = first_image
        # img_model.pic = second_image
        # img_model.merge = third_image
        # img_model.save() #img_model.merge.save('output.jpg', third_image)
        # #######
        # my_dict = {
        #     "name": 'user-name',
        #     "output": img_model,
        # }
        # return render(request,'pages/index2.html',context=my_dict)        
    #MLO end
    return render(request,'pages/upload2.html',context=my_dict)     

###################

def upload(request, argv=None):
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
    ########################
        # image1 = Image.open(images.style)
        # cv2.imshow('mg',image1)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # image_1 = numpy.array(image1)
        # image2 = Image.open(images.pic)
        # image_2 = numpy.array(image2)
    #####
        image_1 = cv2.imread(str(img1)) #modify here # cannot convert to str for filename
        image_2 = cv2.imread(str(img2)) #modify here # cannot convert to str for filename

        # image_1 = cv2.imread(str(list_of_images[0])) #modify here # cannot convert to str for filename
        # image_2 = cv2.imread(str(list_of_images[1])) #modify here # cannot convert to str for filename
        
        #image_2 = cv2.imread(list_of_images[1])#old


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
			# flash("There seems to be more than one face in the images. There must be only one well defined face", "danger")
			# return render_template('main.html', form=form)
        # except Exception as e:
            # print(e)
			# flash("An Unknown Error has occured", "danger")
	        # return render_template('main.html', form=form)
        #print("Done Swapping")

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



####################














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

# def upload_ap(request, argv=None):
#     expected_files = 2
#     my_dict = {"insert_me": "I am from views.py"}

#     if request.method == 'POST':
#         input1 = request.FILES['image1']#.read()#.read() for Type-error-Can't convert object of type 'InMemoryUploadedFile' to 'str' for 'filename'
#         if input1:
#             print('Received1')
#         input2 = request.FILES['image2']#.read()
#         if input2:
#             print('Received2')
#         if not input1 or input2:
#             pass# return redirect('appointment')
#         #MLO
#         images = Customer(style=input1, pic=input2)
        
#         images.save()  

#         list_of_images = [images.style, images.pic]

#         face = base64.b64decode(request.json.get('face'))   
#         face = Image.open(io.BytesIO(face))

#         body = base64.b64decode(request.json.get('body'))
#         body = Image.open(io.BytesIO(body))

#         # convert image to numpy array for processing
#         face = np.array(face)
#         body = np.array(body)

#         # fancy image processing here....
#         face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
#         body_gray = cv2.cvtColor(body, cv2.COLOR_BGR2GRAY)

#         # Create empty matrices in the images' shapes
#         height, width = face_gray.shape
#         mask = np.zeros((height, width), np.uint8)

#         height, width, channels = body.shape

#         # Loading models and predictors of the dlib library to detect landmarks in both faces
#         detector = dlib.get_frontal_face_detector()
#         predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

#         # Getting landmarks for the face that will be swapped into to the body
#         rect = detector(face_gray)[0]

#         # This creates a with 68 pairs of integer values — these values are the (x, y)-coordinates of the facial structures 
#         landmarks = predictor(face_gray, rect)
#         landmarks_points = [] 

    

#         get_landmarks(landmarks, landmarks_points)

#         points = np.array(landmarks_points, np.int32)

#         convexhull = cv2.convexHull(points) 

#         face_cp = face.copy()

#         face_image_1 = cv2.bitwise_and(face, face, mask=mask)

#         rect = cv2.boundingRect(convexhull)

#         subdiv = cv2.Subdiv2D(rect) # Creates an instance of Subdiv2D
#         subdiv.insert(landmarks_points) # Insert points into subdiv
#         triangles = subdiv.getTriangleList()
#         triangles = np.array(triangles, dtype=np.int32)

#         indexes_triangles = []
#         face_cp = face.copy()

    

#     for triangle in triangles :

#         # Gets the vertex of the triangle
#         pt1 = (triangle[0], triangle[1])
#         pt2 = (triangle[2], triangle[3])
#         pt3 = (triangle[4], triangle[5])

#         # Draws a line for each side of the triangle
#         cv2.line(face_cp, pt1, pt2, (255, 255, 255), 3,  0)
#         cv2.line(face_cp, pt2, pt3, (255, 255, 255), 3,  0)
#         cv2.line(face_cp, pt3, pt1, (255, 255, 255), 3,  0)

#         index_pt1 = np.where((points == pt1).all(axis=1))
#         index_pt1 = get_index(index_pt1)
#         index_pt2 = np.where((points == pt2).all(axis=1))
#         index_pt2 = get_index(index_pt2)
#         index_pt3 = np.where((points == pt3).all(axis=1))
#         index_pt3 = get_index(index_pt3)

#         # Saves coordinates if the triangle exists and has 3 vertices
#         if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
#             vertices = [index_pt1, index_pt2, index_pt3]
#             indexes_triangles.append(vertices)

#         # Getting landmarks for the face that will have the first one swapped into
#         rect2 = detector(body_gray)[0]

#         # This creates a with 68 pairs of integer values — these values are the (x, y)-coordinates of the facial structures 
#         landmarks_2 = predictor(body_gray, rect2)
#         landmarks_points2 = []

#         # Uses the function declared previously to get a list of the landmark coordinates
#         get_landmarks(landmarks_2, landmarks_points2)

#         # Generates a convex hull for the second person
#         points2 = np.array(landmarks_points2, np.int32)
#         convexhull2 = cv2.convexHull(points2)

#         body_cp = body.copy()

#         lines_space_new_face = np.zeros((height, width, channels), np.uint8)
#         body_new_face = np.zeros((height, width, channels), np.uint8)

#         height, width = face_gray.shape
#         lines_space_mask = np.zeros((height, width), np.uint8)


#         for triangle in indexes_triangles:

#             # Coordinates of the first person's delaunay triangles
#             pt1 = landmarks_points[triangle[0]]
#             pt2 = landmarks_points[triangle[1]]
#             pt3 = landmarks_points[triangle[2]]

#             # Gets the delaunay triangles
#             (x, y, widht, height) = cv2.boundingRect(np.array([pt1, pt2, pt3], np.int32))
#             cropped_triangle = face[y: y+height, x: x+widht]
#             cropped_mask = np.zeros((height, widht), np.uint8)

#             # Fills triangle to generate the mask
#             points = np.array([[pt1[0]-x, pt1[1]-y], [pt2[0]-x, pt2[1]-y], [pt3[0]-x, pt3[1]-y]], np.int32)
#             cv2.fillConvexPoly(cropped_mask, points, 255)

#             # Draws lines for the triangles
#             cv2.line(lines_space_mask, pt1, pt2, 255)
#             cv2.line(lines_space_mask, pt2, pt3, 255)
#             cv2.line(lines_space_mask, pt1, pt3, 255)

#             lines_space = cv2.bitwise_and(face, face, mask=lines_space_mask)

#             # Calculates the delaunay triangles of the second person's face

#             # Coordinates of the first person's delaunay triangles
#             pt1 = landmarks_points2[triangle[0]]
#             pt2 = landmarks_points2[triangle[1]]
#             pt3 = landmarks_points2[triangle[2]]

#             # Gets the delaunay triangles
#             (x, y, widht, height) = cv2.boundingRect(np.array([pt1, pt2, pt3], np.int32))
#             cropped_mask2 = np.zeros((height,widht), np.uint8)

#             # Fills triangle to generate the mask
#             points2 = np.array([[pt1[0]-x, pt1[1]-y], [pt2[0]-x, pt2[1]-y], [pt3[0]-x, pt3[1]-y]], np.int32)
#             cv2.fillConvexPoly(cropped_mask2, points2, 255)

#             # Deforms the triangles to fit the subject's face : https://docs.opencv.org/3.4/d4/d61/tutorial_warp_affine.html
#             points =  np.float32(points)
#             points2 = np.float32(points2)
#             M = cv2.getAffineTransform(points, points2)  # Warps the content of the first triangle to fit in the second one
#             dist_triangle = cv2.warpAffine(cropped_triangle, M, (widht, height))
#             dist_triangle = cv2.bitwise_and(dist_triangle, dist_triangle, mask=cropped_mask2)

#             # Joins all the distorted triangles to make the face mask to fit in the second person's features
#             body_new_face_rect_area = body_new_face[y: y+height, x: x+widht]
#             body_new_face_rect_area_gray = cv2.cvtColor(body_new_face_rect_area, cv2.COLOR_BGR2GRAY)

#             # Creates a mask
#             masked_triangle = cv2.threshold(body_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
#             dist_triangle = cv2.bitwise_and(dist_triangle, dist_triangle, mask=masked_triangle[1])

#             # Adds the piece to the face mask
#             body_new_face_rect_area = cv2.add(body_new_face_rect_area, dist_triangle)
#             body_new_face[y: y+height, x: x+widht] = body_new_face_rect_area

#             body_face_mask = np.zeros_like(body_gray)
#             body_head_mask = cv2.fillConvexPoly(body_face_mask, convexhull2, 255)
#             body_face_mask = cv2.bitwise_not(body_head_mask)

#             body_maskless = cv2.bitwise_and(body, body, mask=body_face_mask)
#             result = cv2.add(body_maskless, body_new_face)

#             # Gets the center of the face for the body
#             (x, y, widht, height) = cv2.boundingRect(convexhull2)
#             center_face2 = (int((x+x+widht)/2), int((y+y+height)/2))

#             new = cv2.seamlessClone(result, body, body_head_mask, center_face2, cv2.NORMAL_CLONE)
            
#             # done fancy processing....

            
#             # build a response to send back to client
#             new = Image.fromarray(new)
#             byte_arr = io.BytesIO()
#             new.save(byte_arr, format='PNG') # convert the PIL image to byte array
#             encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
            
#             return {'image': encoded_img}

#         # my_dict = {
#         #     "name": 'user-name',
#         #     "output": img_model,
#         # }
#         return render(request,'pages/output.html')#,context=my_dict)        





################Firebase###############
from django.conf import settings
from django.core.files.storage import default_storage
#from django.contrib import messages
import pyrebase
import os
import urllib

config = {
    "apiKey": "AIzaSyAFrdm_BG3qw9sWJhHqBEDh_7k1ZSzBizI",
    "authDomain": "faceswap-d9f07.firebaseapp.com",
    "projectId": "faceswap-d9f07",
    "storageBucket": "faceswap-d9f07.appspot.com",
    "messagingSenderId": "211027390522",
    "appId": "1:211027390522:web:5b760a060095585c6f0503",
    "measurementId": "G-Z393E4W8ZZ",
    "databaseURL": "",
}

firebase = pyrebase.initialize_app(config)
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

# class StyleCreateView(generic.CreateView):
#     model = Style
#     form_class = forms.StyleForm
#     template_name = 'pages/style_create_form.html'

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