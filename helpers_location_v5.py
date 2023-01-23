import cv2
import numpy as np
import matplotlib.pyplot as plt
import re
from statistics import mode,mean,median
import pandas as pd

import random

IMG_WIDTH = 4000
IMG_HEIGHT = 6500
#parameters for linking the within-colunm rows.
PARA_WITHIN_COLUMN = [30,100,50]
#parameters for linking the partitions.
# 20 3000 300
PARA_PARTITION_1 = [20,2000,300]
PARA_PARTITION_2 = 500
# v4 parameters for 4000*6500 pictures.
PARA_V4 = [50,20,80]
# Threshold for image thresholding. 200 seems to be good for a faint image.
THRESHOLD_1 = 200

def show_plot(img,size = (15,15)):
    plt.figure(figsize=size)
    plt.imshow(img)
    plt.show()

def show_anno(IMAGES_PATH,filename,anno_groups,x_sorted_annotations):
    newImg = cv2.imread(IMAGES_PATH + "/" + filename)
    #for column in groups:


    for group in anno_groups:
        color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        for i in group:
        #if word.bounding_poly.vertices[2].x< second-20 or word.bounding_poly.vertices[0].x > third-20:
            #continue
            pts = np.array([[[vertex.x, vertex.y] for vertex in x_sorted_annotations[i].bounding_poly.vertices]], np.int32)
            cv2.polylines(newImg, [pts], True, color, 5)

    show_plot(newImg)

def text_density(x_sorted_annotations, y_sorted_annotations):
    ## Find total page area
    y_top = y_sorted_annotations[0].bounding_poly.vertices[0].y
    y_bottom = y_sorted_annotations[-1].bounding_poly.vertices[2].y
    x_left = x_sorted_annotations[0].bounding_poly.vertices[0].x
    x_right = x_sorted_annotations[-1].bounding_poly.vertices[2].x
    page_area = (y_bottom - y_top)*(x_right - x_left)

    ## Iterates through each annotation 
    total_anno_area = 0
    for annotation in y_sorted_annotations:
        y_top = annotation.bounding_poly.vertices[0].y
        y_bottom = annotation.bounding_poly.vertices[2].y
        x_left = annotation.bounding_poly.vertices[0].x
        x_right = annotation.bounding_poly.vertices[2].x
        anno_area = (y_bottom - y_top)*(x_right - x_left)
        total_anno_area += anno_area
    return total_anno_area/page_area
    
    
def get_number(filename):
    i, j = filename.index("("), filename.index(")")
    image_number = int(filename[i+1:j])
    if filename.endswith('1L.jpg') or filename.endswith('1L.tif'):
        page = "1L"
    else:
        page = "2R"

    return image_number, page

def get_country_page(x_sorted_annotations,img):
    width = img.shape[1]
    possible_countries = []
    bad_words = ["NAME", "BANK", "LONDON", "PLACE","PLA","PLAOR", "OFFICE", "CORRES","PONDENT","SUB","AGENCY","BRANCH","COMMEND","AGENT","OF","ALMAN","YEAR","BOOK","AUXILIARY"]
    capital_annotations = []

    # Get possible page
    page = ''
    for i,anno in enumerate(x_sorted_annotations):
        possible_page = re.compile(r'\d+').findall(anno.description)
        if len(possible_page) == 1 and len(possible_page[0]) == 4:
            page = possible_page[0]
            continue

    # Get Country
    idx_possible_counrty = []
    for i,anno in enumerate(x_sorted_annotations):
        # 0.12 used to be 0.18
        if anno.bounding_poly.vertices[0].x > 0.12* width and anno.bounding_poly.vertices[1].x < 0.85 * width:
            country = anno.description
            up_count = 0
            for letter in country:
                if letter.isupper():
                    up_count += 1
            if up_count/len(country) > 0.7 and len(country) >2 :
                capital_annotations.append(i)
                bad = False
                for word in bad_words:
                    if word in country:
                        bad = True
                if bad == False:
                    idx_possible_counrty.append(i)
    
    idxs_appended = []
    for idx1 in idx_possible_counrty:
        if idx1 in idxs_appended:
            continue
        x1 = x_sorted_annotations[idx1].bounding_poly.vertices[1].x
        y1 = x_sorted_annotations[idx1].bounding_poly.vertices[1].y
        possible_country = [idx1]

        for idx2 in idx_possible_counrty:
            x2 = x_sorted_annotations[idx2].bounding_poly.vertices[0].x
            y2 = x_sorted_annotations[idx2].bounding_poly.vertices[0].y
            if abs(y1-y2) < 50 and  0<x2-x1<180:
                x1 = x_sorted_annotations[idx2].bounding_poly.vertices[1].x
                y1 = x_sorted_annotations[idx2].bounding_poly.vertices[1].y
                possible_country.append(idx2)
                idxs_appended.append(idx2)
        
        x_1 = x_sorted_annotations[possible_country[0]].bounding_poly.vertices[0].x
        y_1 = x_sorted_annotations[possible_country[0]].bounding_poly.vertices[0].y
        x_2 = x_sorted_annotations[possible_country[-1]].bounding_poly.vertices[1].x
        y_2 = x_sorted_annotations[possible_country[-1]].bounding_poly.vertices[1].y
        for i, anno in enumerate(x_sorted_annotations):
            if anno.description == "(":
                x_brace = anno.bounding_poly.vertices[1].x
                y_brace = anno.bounding_poly.vertices[1].y
                if abs(y_1-y_brace) < 50 and  -10 <x_1-x_brace<50:
                    possible_country.insert(0, i)
                    continue
        for j, anno in enumerate(x_sorted_annotations):
            if anno.description == ")":
                x_brace = anno.bounding_poly.vertices[0].x
                y_brace = anno.bounding_poly.vertices[0].y
                if abs(y_2-y_brace) < 50 and  0<x_brace-x_2<50:
                    possible_country.append(j) 
                    continue       
        if len(possible_country) == 1 and x_sorted_annotations[possible_country[0]].description == "AND":
            continue
        possible_countries.append(possible_country)
    
    possible_countries = sorted(possible_countries, key= lambda x:x_sorted_annotations[x[0]].bounding_poly.vertices[0].y)

    # Clean banks:
    cleaned_possible_countries = []
    '''
    for country in possible_countries:
        des = ""
        for idx in country:
            des+= x_sorted_annotations[idx].description
        if not("BANK" in des):
            cleaned_possible_countries.append(country)'''
    length_list = []
    for anno in x_sorted_annotations:
        length_list.append(anno.bounding_poly.vertices[3].y-anno.bounding_poly.vertices[0].y)
    avg_length = median(length_list)
    for country in possible_countries:
        length = x_sorted_annotations[country[0]].bounding_poly.vertices[3].y - x_sorted_annotations[country[0]].bounding_poly.vertices[0].y
        if length > 1.3*avg_length:
            cleaned_possible_countries.append(country)

    countries = []
    added = []
    for i, possible_country_1 in enumerate(cleaned_possible_countries):
        if i in added:
            continue
        y1 = x_sorted_annotations[possible_country_1[0]].bounding_poly.vertices[2].y
        for j, possible_country_2 in enumerate(cleaned_possible_countries):
            y2 = x_sorted_annotations[possible_country_2[0]].bounding_poly.vertices[0].y
            if 0<y2-y1<100:
                possible_country_1 += possible_country_2
                y1 = x_sorted_annotations[possible_country_2[0]].bounding_poly.vertices[2].y
                added.append(j)
        countries.append(possible_country_1)

    countries_height = []
    for country in countries:
        height_1 = x_sorted_annotations[country[0]].bounding_poly.vertices[0].y
        height_2 = x_sorted_annotations[country[-1]].bounding_poly.vertices[0].y
        countries_height.append((height_1,height_2))

    # What if country is not detected

    return countries, countries_height, capital_annotations, page

def show_countries(countries,x_sorted_annotations):
    for country in countries:
        country_des = ""
        for i in country:
            country_des += " "+x_sorted_annotations[i].description
        print(country_des)

def anno_num_detect(x_sorted_annotations):
    columns =[ [],[],[] ]
    for i,anno in enumerate(x_sorted_annotations):
        # Column 1
        word = anno.description
        x = anno.bounding_poly.vertices[0].x
        x_list = []
        column = 0
        if "PLAC" in word or "LACE" in word:
            x_list = [x,x-100,x-200]
            column = 1
        if word in "NAME" or word in "OF" or word in "BANK":
            x_list = [x,x-100,x-150]
            column = 2
        if word in "LONDON" or word in "OFFICE" or "CORRESP" in word:
            x_list = [x,x-100,x-150]
            column = 3
        if column == 0:
            continue
        
        for x_2 in x_list:
            num = 0
            for j,anno_2 in enumerate(x_sorted_annotations):
                if anno_2.bounding_poly.vertices[0].x < x_2 and anno_2.bounding_poly.vertices[1].x > x_2:
                    num += 1
            columns[column-1].append(num)
        
    num_list = [0,0,0]
    for i in [0,1,2]:
        if len(columns[i]) != 0:
            num_list[i] = max(columns[i])
    return num_list

def column_judge(img_shape,line_m,line_r,x,y):
    # Partition: Ax+By+C = 0, where A = y2-y1; B = x1-x2; C = x2y1-x1y2
    # For a point (x1, y1), if sign(Ax1+By1+C) = sign(A*0+B*(img.y/2)+C), then on the left of the partition.
    x1_m, y1_m, x2_m, y2_m = line_m[0]
    x1_r, y1_r, x2_r, y2_r = line_r[0]

    # For ~
    x1_m, x2_m, x1_r, x2_r = x1_m-10, x2_m-10, x1_r-10, x2_r-10
    anno_line_m = np.sign((y2_m-y1_m) * x + (x1_m-x2_m) * y + x2_m * y1_m-x1_m * y2_m)
    anno_line_r = np.sign((y2_r-y1_r) * x + (x1_r-x2_r) * y + x2_r * y1_r-x1_r * y2_r)

    left_line_m = np.sign((x1_m-x2_m) * img_shape[0]/2 + x2_m * y1_m-x1_m * y2_m)
    left_line_r = np.sign((x1_r-x2_r) * img_shape[0]/2 + x2_r * y1_r-x1_r * y2_r)

    type = 0
    if anno_line_m == left_line_m and anno_line_r == left_line_r:
        type = 0
    elif anno_line_m == -left_line_m and anno_line_r == left_line_r:
        type = 1
    elif anno_line_m == -left_line_m and anno_line_r == -left_line_r:
        type = 2
    
    return type

def draw_anno_bounding(img, x_sorted_annotations, mode, color = (0,0,0), spread_x = 0, spread_y = 0):
    if mode == "mid":
        left_1,left_2 = 0, 3
        right_1,right_2 = 1, 2
    if mode == "top":
        left_1,left_2 = 0, 0
        right_1,right_2 = 1, 1
    if mode == "bot":
        left_1,left_2 = 3, 3
        right_1,right_2 = 2, 2
    for anno in x_sorted_annotations:        
        left_x = int(0.5*(anno.bounding_poly.vertices[left_1].x + anno.bounding_poly.vertices[left_2].x))
        left_y = int(0.5*(anno.bounding_poly.vertices[left_1].y + anno.bounding_poly.vertices[left_2].y))
        right_x = int(0.5*(anno.bounding_poly.vertices[right_1].x + anno.bounding_poly.vertices[right_2].x))
        right_y = int(0.5*(anno.bounding_poly.vertices[right_1].y + anno.bounding_poly.vertices[right_2].y))
        cv2.line(img,(left_x-spread_x,left_y+spread_y),(right_x+spread_x,right_y+spread_y),color, 5)   

def v5_grouping_column(img_shape,x_sorted_annotations, alte_partition, para_1 = PARA_WITHIN_COLUMN, para_2 = PARA_PARTITION_1, para_3 = PARA_PARTITION_2):
    """ Group the rows within columns using computer vision.

        @param img_shape (List[int,int]): width and height of the orginal image.
    
        @returns 
    """
    error_partition = False
    img_group_column = np.uint8(np.full((img_shape[0],img_shape[1],3),255))

    # Link the two mid-points of the annotation boxes.
    draw_anno_bounding(img_group_column, x_sorted_annotations, mode = "mid", color = (0,0,0), spread_x = 0, spread_y = 0)

    imgGray_1=cv2.cvtColor(img_group_column,cv2.COLOR_BGR2GRAY)
    imgBW_1=cv2.threshold(imgGray_1, 150, 255, cv2.THRESH_BINARY_INV)[1]
    Lines= cv2.HoughLinesP(imgBW_1,1,np.pi/180,para_1[0],minLineLength=para_1[1],maxLineGap=para_1[2])
    # We may want to store the grouped column-level rows.

    for line in Lines:
        x1,y1,x2,y2 = line[0]
        if x2 != x1:
                # if abs(y2-y1)/abs(x2-x1)<0.3:
            if abs(y2-y1)/abs(x2-x1)<0.8:
                cv2.line(img_group_column,(x1,y1),(x2,y2),(0,0,0),5)
    #show_plot(img_group_column)

    imgGray_2=cv2.cvtColor(img_group_column,cv2.COLOR_BGR2GRAY)
    imgBW_2=cv2.threshold(imgGray_2, 230, 255, cv2.THRESH_BINARY_INV)[1]
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(imgBW_2)

    # Store the left points of the lines to generate partitions.
    img_lp = np.uint8(np.full((img_shape[0],img_shape[1],3),255))
    idx = 0

    anno_mid_points = [ [i,int(np.average([anno.bounding_poly.vertices[i].x for i in [0,1,2,3]])),int(np.average([anno.bounding_poly.vertices[i].y for i in [0,1,2,3]])),anno.description] for i,anno in enumerate(x_sorted_annotations)]

    groups = [[] for _ in range(n)]

    for i,x_mid,y_mid,des in anno_mid_points:
        label = labels[y_mid,x_mid]
        groups[label].append(i)

    #groups.append(group)

    for group in groups:
        if group == []: continue
        x = 0.5 * (x_sorted_annotations[group[0]].bounding_poly.vertices[0].x + x_sorted_annotations[group[0]].bounding_poly.vertices[3].x)
        y = 0.5 * (x_sorted_annotations[group[0]].bounding_poly.vertices[0].y + x_sorted_annotations[group[0]].bounding_poly.vertices[3].y)
        cv2.circle(img_lp,(int(x), int(y)),7,(0,0,0),cv2.FILLED)


    img_lp = np.uint8(img_lp)
    imgGray_3=cv2.cvtColor(img_lp,cv2.COLOR_BGR2GRAY)
    imgBW_3=cv2.threshold(imgGray_3, 230, 255, cv2.THRESH_BINARY_INV)[1]

    lines = cv2.HoughLinesP(imgBW_3,0.5,np.pi/360,para_2[0],minLineLength=para_2[1],maxLineGap=para_2[2])
    # if there are multiple countries in the image, we may want to increase the maxLineGap
    if lines is None:
        lines = cv2.HoughLinesP(imgBW_3,0.5,np.pi/360,20,para_2[0],minLineLength=para_2[1]+1000,maxLineGap=para_2[2]*2)
    if lines is None:
        lines = [[[200,400,200,1000]],[[1200,400,1200,1000]],[[2200,400,2200,1000]]]
    lines = sorted (lines, key=lambda x: x[0][0])

        #line_l, line_m, line_r= lines[0],lines[int(len(lines)/3)],lines[-1]
    x_l,x_m,x_r = alte_partition[0][0],alte_partition[1][0],alte_partition[2][0]
    line_l = np.array([[x_l,50,x_l,2000]])
    line_m = np.array([[x_m,50,x_m,2000]])
    line_r = np.array([[x_r,50,x_r,2000]])
    if line_r[0][0] - line_m[0][0] < para_3 or line_m[0][0] - line_l[0][0] < para_3:
        error_partition = True
    # Use the jump of x-value to seperate partitions.
    jump = [i for i,line in enumerate(lines) if i>0 and line[0][0]>lines[i-1][0][0]+para_3]
    if 1000<lines[0][0][0]<2000 and len(jump) == 1:
        line_m = max(lines[:jump[0]], key=lambda x: abs(x[0][0]-x[0][3]))
        line_r = max(lines[jump[0]:], key=lambda x: abs(x[0][0]-x[0][3]))
    if len(jump) == 2:
        line_l = max(lines[0:jump[0]], key=lambda x: abs(x[0][0]-x[0][3]))
        line_m = max(lines[jump[0]:jump[1]], key=lambda x: abs(x[0][0]-x[0][3]))
        line_r = max(lines[jump[1]:], key=lambda x: abs(x[0][0]-x[0][3]))
    '''else:
        #line_l, line_m, line_r= lines[0],lines[int(len(lines)/3)],lines[-1]
        x_l,x_m,x_r = alte_partition[0][0],alte_partition[1][0],alte_partition[2][0]
        line_l = np.array([[x_l,50,x_l,2000]])
        line_m = np.array([[x_m,50,x_m,2000]])
        line_r = np.array([[x_r,50,x_r,2000]])
        if line_r[0][0] - line_m[0][0] < para_3 or line_m[0][0] - line_l[0][0] < para_3:
            error_partition = True'''

    '''
    #img_show = np.uint8(np.full((img_shape[0],img_shape[1],3),255))
    for line in [line_l,line_m,line_r]:
        x1,y1,x2,y2 = line[0]

        cv2.line(img_group_column,(x1,y1),(x2,y2),(0,0,0),2)
    show_plot(img_group_column)'''

    anno_type = []
    for i,x,y,des in anno_mid_points:
        type = column_judge(img_shape,line_m,line_r,x+20,y)
        anno_type.append((i,type))

    rows_c1, rows_c2,rows_c3 = [],[],[] 
    #rows_c1_position, rows_c2_position,rows_c3_position = [],[],[]
    for group in groups[1:]:
        row_c1, row_c2,row_c3 = [],[],[]
        #row_c1_position, row_c2_position,row_c3_position = [],[],[]
        for i in group:
            if anno_type[i][1] == 0:
                row_c1.append(i)
            if anno_type[i][1] == 1:
                row_c2.append(i)
            if anno_type[i][1] == 2:
                row_c3.append(i)
        if row_c1 != []: 
            rows_c1.append(row_c1)
            #rows_c1_position.append([row_c1[0],row_c1[0],row_c1[0],row_c1[0]])
        if row_c2 != []: rows_c2.append(row_c2)
        if row_c3 != []: rows_c3.append(row_c3)
    v5_groups = [rows_c1,rows_c2,rows_c3]

    return line_l,line_m,line_r,error_partition, v5_groups


def grouping_fix_failure(groups,x_sorted_annotations,capital_annotations):
    new_groups = [[],[],[]]
    for column in [0,1,2]:
        groups_idx = list(zip(groups[column],range(len(groups[column]))))
        x_sorted_groups_idx = sorted(groups_idx, key=lambda x: x_sorted_annotations[x[0][0]].bounding_poly.vertices[0].x)
        
        #group the failure
        idxs_appended = []
        groups_new = []
        for i, group1_idx in enumerate(x_sorted_groups_idx):
            group1 = group1_idx[0]
            idx1 = group1_idx[1]
            if idx1 in idxs_appended:
                continue
            x1 = 0.5 * (x_sorted_annotations[group1[0]].bounding_poly.vertices[0].x + x_sorted_annotations[group1[0]].bounding_poly.vertices[3].x)
            y1 = 0.5 * (x_sorted_annotations[group1[0]].bounding_poly.vertices[0].y + x_sorted_annotations[group1[0]].bounding_poly.vertices[3].y)
            x2 = 0.5 * (x_sorted_annotations[group1[-1]].bounding_poly.vertices[1].x + x_sorted_annotations[group1[-1]].bounding_poly.vertices[2].x)
            y2 = 0.5 * (x_sorted_annotations[group1[-1]].bounding_poly.vertices[1].y + x_sorted_annotations[group1[-1]].bounding_poly.vertices[2].y)
            #print(x1,y1,x2,y2,x_sorted_annotations[group1[0]].description,x_sorted_annotations[group1[-1]].description)
            if idx1 > 3: 
                iter_groups_idx = groups_idx[idx1-4:idx1]+groups_idx[idx1+1:idx1+5]
                iter_groups_idx = sorted(iter_groups_idx, key= lambda x:x_sorted_annotations[x[0][0]].bounding_poly.vertices[0].x)
            if idx1 < 4: 
                iter_groups_idx = groups_idx[idx1+1:idx1+5]
                iter_groups_idx = sorted(iter_groups_idx, key= lambda x:x_sorted_annotations[x[0][0]].bounding_poly.vertices[0].x)
            for j,group2_idx in enumerate(iter_groups_idx):
                group2 = group2_idx[0]
                idx2 = group2_idx[1]
                x_1 = 0.5 * (x_sorted_annotations[group2[0]].bounding_poly.vertices[0].x + x_sorted_annotations[group2[0]].bounding_poly.vertices[3].x)
                y_1 = 0.5 * (x_sorted_annotations[group2[0]].bounding_poly.vertices[0].y + x_sorted_annotations[group2[0]].bounding_poly.vertices[3].y)
                x_2 = 0.5 * (x_sorted_annotations[group2[-1]].bounding_poly.vertices[1].x + x_sorted_annotations[group2[-1]].bounding_poly.vertices[2].x)
                y_2 = 0.5 * (x_sorted_annotations[group2[-1]].bounding_poly.vertices[1].y + x_sorted_annotations[group2[-1]].bounding_poly.vertices[2].y)
                if (abs(y2-y_1)<50 and 0< (x_1 - x2)< 80) or (abs(y2-y_1)<30 and 0< (x_1 - x2)< 160) :
                    #x_sorted_groups_idx[i][0] = x_sorted_groups_idx[i][0]+ group2_idx[0]
                    group1 = group1 + group2
                    idxs_appended.append(idx2)
                    x2,y2 = x_2,y_2
            
            capital = False
            for idx in capital_annotations:
                if idx in group1:
                    capital = True
                    break
            if capital == False:
                groups_new.append(group1)
        groups_new = sorted(groups_new, key = lambda x: x_sorted_annotations[x[0]].bounding_poly.vertices[0].y)

        new_groups[column] = groups_new

    # Fix multi-row bank names:

    new_bank_name = []
    for i, row in enumerate(new_groups[1]):
        if i<3:
            new_bank_name.append(row)
            x_store = x_sorted_annotations[row[0]].bounding_poly.vertices[0].x
            if "NAME" in x_sorted_annotations[row[0]].description:
                new_bank_name = []
            continue
        if new_bank_name == []:
            new_bank_name.append(row)
            x_store = x_sorted_annotations[row[0]].bounding_poly.vertices[0].x
        else:
            x0 = x_sorted_annotations[row[0]].bounding_poly.vertices[0].x
            if x0 - x_store > 45:
                new_bank_name[-1] += row
                x_store = 10000
            else:
                descrip = ""
                for i in row:
                    descrip += x_sorted_annotations[i].description
                if "Sub" in descrip or "Auxiliary" in descrip:
                    continue
                
                new_bank_name.append(row)
                x_store = x_sorted_annotations[row[0]].bounding_poly.vertices[0].x                

    # Fix curly braces:
    
    new_place = []
    for i, row in enumerate(new_groups[0]):
        x0 = x_sorted_annotations[row[0]].bounding_poly.vertices[0].x
        #if x0 < 200: continue
        if i<3:
            new_place.append(row)
            x_store = x_sorted_annotations[row[0]].bounding_poly.vertices[0].x
            if "PLA" in x_sorted_annotations[row[0]].description:
                new_place = []
            continue
        if new_place == []:
            new_place.append(row)
            x_store = x_sorted_annotations[row[0]].bounding_poly.vertices[0].x
        else:
            if x0 - x_store > 110:
                # Can be used to calculate branch number
                continue
            else:
                descrip = ""
                for i in row:
                    descrip += x_sorted_annotations[i].description
                if len(descrip) < 4:
                    continue
                if "Sub" in descrip or "Agency" in descrip or "Agoncy" in descrip:
                    continue
                if ">" in x_sorted_annotations[row[0]].description or "''" in x_sorted_annotations[row[0]].description:
                    continue
                new_place.append(row)
                x_store = x_sorted_annotations[row[0]].bounding_poly.vertices[0].x 

    new_groups[0] = new_place
    new_groups[1] = new_bank_name
    return new_groups


def v4_grouping_row(img,x_sorted_annotations,threshold = THRESHOLD_1, para_1 = PARA_V4,show = False):

    imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBW=cv2.threshold(imgGray, threshold, 255, cv2.THRESH_BINARY_INV)[1]
    # This parameter tbd.
    slice_height = 500
    slice_num = img.shape[0]//slice_height
    
    # Crop and run v4.
    for i in range(slice_num):
        img_crop = imgBW[slice_height*i:slice_height*(i+1),:]
        imgLines= cv2.HoughLinesP(img_crop,1,np.pi/10,para_1[0],minLineLength=para_1[1],maxLineGap=para_1[2])
        if not imgLines is None:
            for line in imgLines:
                x1,y1,x2,y2 = line[0]
                if x2 != x1:
                    if abs(y2-y1)/abs(x2-x1)<0.3:
                        cv2.line(img,(x1,y1+slice_height*i),(x2,y2+slice_height*i),(0,0,0),3)

    # Optional: Dilate
    kernal = np.ones((5,5),np.uint8)
    img = cv2.dilate(img, kernal, iterations = 1)


    # Seperate the groups with cv2.
    draw_anno_bounding(img, x_sorted_annotations, mode = "top", color = (255,255,255), spread_x = 7, spread_y = 0)
    
    # Optional: draw a line under the box.
    draw_anno_bounding(img, x_sorted_annotations, mode = "bot", color = (255,255,255), spread_x = 0, spread_y = 3)

    
    imgGray_2=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBW_2=cv2.threshold(imgGray_2, 210, 255, cv2.THRESH_BINARY_INV)[1]
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(imgBW_2)

    # Store the groups:
    v4_groups_idx = [i for i in range(1,n) if stats[i][2] > 100 and stats[i][2] < imgBW_2.shape[1]*0.95]

    v4_groups_idx = sorted (v4_groups_idx, key = lambda x: centroids[x][1])


    
    # Optional 1: Cover Rate.
    # Fingers and shadows should be cleaned.
    #mask = np.full([img_grouped.shape[0],img_grouped.shape[1]],False)
    if show == True:
    # Optional 2: Outut Store.
        img_grouped = np.full((imgBW_2.shape[0],imgBW_2.shape[1],3),255)
        for i in range(1,n):
            # 300000
            if stats[i][2] > 100 and stats[i][2] < imgBW_2.shape[1]*0.95:
                mask = labels == i
                img_grouped[:,:,0][mask] = np.random.randint(0,255)
                img_grouped[:,:,1][mask] = np.random.randint(0,255)
                img_grouped[:,:,2][mask] = np.random.randint(0,255)
        show_plot(img_grouped)
    #plt.figure(figsize=(20,20))
    #plt.imshow(img_grouped)
    #plt.show()
    #print(groups)

    return v4_groups_idx, labels

def v5_get_rows(x_sorted_annotations, v4_groups_idx, labels, v5_groups,img_shape):

    # v4_groups_idx: the group number of v4, say, [1,3,4,5,6,7,9] needed???
    # Let's see v4 grouped what v5 groups together.
    labels_columns = []
    for column in v5_groups:
        labels_column = []
        for row in column:
            row_labels = []
            for idx in row:
                word_labels = []
                vert = x_sorted_annotations[idx].bounding_poly.vertices
                x0,y0 = vert[0].x, vert[0].y
                x1,y1 = vert[1].x, vert[1].y
                x2,y2 = vert[2].x, vert[2].y
                x3,y3 = vert[3].x, vert[3].y
                weight = [0.1,0.2,0.3,0.4,0.5]
                for w1 in weight:
                    x = int(w1 * x0 + (1-w1) * x1)
                    if x >= img_shape[1]:
                        x = img_shape[1]-1
                    for w2 in weight:
                        y = int(w2 * y0 + (1-w2) * y2)
                        if y >= img_shape[0]:
                            y = img_shape[0]-1
                        if labels[y][x] != 0:
                            word_labels.append(labels[y][x])
                if len(word_labels) >0:
                    word_label = mode(word_labels)
                else:
                    word_label = None
                row_labels.append(word_label)
            row_label = mode(row_labels)
            labels_column.append(row_label)
        labels_columns.append(labels_column)

    # v5_groups
    rows = []
    for i,bank_idxs in enumerate(v5_groups[1]):
        row = []
        label_1 = labels_columns[1][i]
        y_bank_1 = x_sorted_annotations[v5_groups[1][i][0]].bounding_poly.vertices[0].y
        y_bank_2 = x_sorted_annotations[v5_groups[1][i][-1]].bounding_poly.vertices[1].y
        if not(label_1 is None):
            places = []
            for j,place_idx in enumerate(v5_groups[0]):
                label_0 = labels_columns[0][j]
                if label_1 == label_0:
                    places.append(j)
            if len(places) == 0:
                place = None
            if len(places) == 1:
                place = places[0]
            if len(places) >1:
                place = None
                distance_store = 250
                for idx_place in places:
                    distance = abs(x_sorted_annotations[v5_groups[0][idx_place][-1]].bounding_poly.vertices[1].y - y_bank_1)
                    if distance < distance_store:
                        distance_store = distance
                        place = idx_place

            offices = []
            for j,office_idx in enumerate(v5_groups[2]):
                label_2 = labels_columns[2][j]
                if label_1 == label_2:
                    offices.append(j)
            if len(offices) == 0:
                office = None
            if len(offices) == 1:
                office = offices[0]
            if len(offices) >1:
                office = None
                distance_store = 250
                for idx_office in offices:
                    distance = abs(x_sorted_annotations[v5_groups[2][idx_office][0]].bounding_poly.vertices[0].y - y_bank_2)
                    if distance < distance_store:
                        distance_store = distance
                        office = idx_office      

            row = [place, i, office]

            
        else:
            row = [None,i, None]
        
        rows.append(row)
            
    return rows



def rows_fix_failure(rows,v5_groups,x_sorted_annotations):
    # Get a added list
    c1_added = set()
    c2_added_c1 = set()
    c2_added_c3 = set()
    c3_added = set()
    for i,row in enumerate(rows): 
        if not row[0] is None:
            c1_added.add(row[0])
            c2_added_c1.add(row[1])
        if not row[2] is None:
            c3_added.add(row[2])
            c2_added_c3.add(row[1])

    for idx_c1 in range(len(v5_groups[0])):
        if idx_c1 in c1_added:
            continue
        x_c1 = x_sorted_annotations[v5_groups[0][idx_c1][-1]].bounding_poly.vertices[2].x
        y_c1 = x_sorted_annotations[v5_groups[0][idx_c1][-1]].bounding_poly.vertices[2].y
        #idx_c2_matched = -1
        dis_store = 150
        c2_matchc1 = None
        for idx_c2 in range(len(v5_groups[1])):
            if idx_c2 in c2_added_c1:
                continue
            
            y_c2 = x_sorted_annotations[v5_groups[1][idx_c2][0]].bounding_poly.vertices[3].y            
            if abs(y_c2-y_c1)<dis_store:
                dis_store=abs(y_c2-y_c1)
                c2_matchc1 = idx_c2

        if c2_matchc1 is None:
            continue

        word = ''
        for idx in v5_groups[0][idx_c1]:
            word += x_sorted_annotations[idx].description
        
        if len(word)<4:
            continue

        up_count = 0
        dot_list = [".",",","#","·","…","*"]
        for letter in word:
            if letter in dot_list:
                up_count += 1
        if up_count/len(word) > 0.7:
            continue

        if abs(x_c1-x_sorted_annotations[v5_groups[1][c2_matchc1][0]].bounding_poly.vertices[3].x)> 800:
            continue


        rows[c2_matchc1][0] = idx_c1
        c2_added_c1.add(c2_matchc1)

    for idx_c3 in range(len(v5_groups[2])):
        if idx_c3 in c3_added:
            continue
        x_c3 = x_sorted_annotations[v5_groups[2][idx_c3][0]].bounding_poly.vertices[3].x
        y_c3 = x_sorted_annotations[v5_groups[2][idx_c3][0]].bounding_poly.vertices[3].y
        #idx_c2_matched = -1
        dis_store = 150
        c2_matchc3 = None
        for idx_c2 in range(len(v5_groups[1])):
            if idx_c2 in c2_added_c3:
                continue
            
            y_c2 = x_sorted_annotations[v5_groups[1][idx_c2][-1]].bounding_poly.vertices[2].y            
            if abs(y_c2-y_c3)<dis_store:
                dis_store=abs(y_c2-y_c3)
                c2_matchc3 = idx_c2

        if c2_matchc3 is None:
            continue
        
        #if too short; too far away in x; too many capital, no
        word = ''
        for idx in v5_groups[2][idx_c3]:
            word += x_sorted_annotations[idx].description
        
        if len(word)<5:
            continue

        up_count = 0
        for letter in word:
            if letter.isupper():
                up_count += 1
        if up_count/len(word) > 0.7 and len(word) >2 :
            continue

        if abs(x_c3-x_sorted_annotations[v5_groups[1][c2_matchc3][-1]].bounding_poly.vertices[2].x)> 800:
            continue

        rows[c2_matchc3][2] = idx_c3
        c2_added_c3.add(c2_matchc3)
    num_c1,num_c2,num_c3 = 0,0,0
    for row in rows:
        if not row[0] is None:
            num_c1 += 1
        if not row[1] is None:
            num_c2 += 1
        if not row[2] is None:
            num_c3 += 1    
    num_v5 = [num_c1,num_c2,num_c3]        
    return rows,num_v5

def rows_clean_dots(rows,v5_groups,x_sorted_annotations):
    for i,row in enumerate(rows):
        dot_list = [".",",","#","·","…","*"]
        # Plcae:
        if row[0] is None:
            pass
        else:
            place_list = v5_groups[0][row[0]]
            if place_list == []:
                pass
            else:
                length = len(place_list)

                for i in range(7):
                    des = x_sorted_annotations[place_list[-1]].description
                    up_count = 0
                    for letter in des:
                        if letter in dot_list:
                            up_count += 1
                    if up_count/len(des) > 0.7 :#or (u'\u4e00' <= letter <= u'\u9fff'):
                        v5_groups[0][row[0]].pop(-1)
                        if length == i+1:
                            break
                        
                    else:
                        break
        # Bank Name
        bankname_list = v5_groups[1][row[1]]
        length = len(bankname_list)
        for i in range(7):
            des = x_sorted_annotations[bankname_list[-1]].description
            up_count = 0
            for letter in des:
                if letter in dot_list:
                    up_count += 1
            if up_count/len(des) > 0.7:
                v5_groups[1][row[1]].pop(-1)
                if length == i+1:
                    break
                
            else:
                break        
    return v5_groups


def gen_csv(countries,countries_height,rows,v5_groups,x_sorted_annotations):
    csv_countries = []
    for i,country in enumerate(countries):
        csv_country =[]
        rows_height = []
        country_name = ""
        dis_row = 0
        dis = []

        for idx in country:
            country_name += " "+x_sorted_annotations[idx].description.capitalize()
        country_name = country_name[1:]
        for l,row in enumerate(rows):
            if v5_groups[1][row[1]] == []:
                continue
            row_height = x_sorted_annotations[v5_groups[1][row[1]][0]].bounding_poly.vertices[0].y
            rows_height.append(row_height)

            if l > 0:
                if len(rows_height)>1:
                    dis.append(rows_height[-1]-rows_height[-2])
            

            
            if i+1 == len(countries):
                if row_height < countries_height[i][1]:
                    continue
            else:
                if row_height < countries_height[i][1] or row_height > countries_height[i+1][0]:
                    continue
            
            place = ""
            bank_name = ""
            office = ""
            if not row[0] is None:
                if v5_groups[0][row[0]] == []:
                    pass
                else:
                    for j in v5_groups[0][row[0]]:
                        place += " "+x_sorted_annotations[j].description
                    place = place[1:]
            if not row[1] is None:
                for j in v5_groups[1][row[1]]:
                    bank_name += " "+x_sorted_annotations[j].description
                bank_name = bank_name[1:]
            if not row[2] is None:
                if v5_groups[2][row[2]] == []:
                    pass
                else:
                    for j in v5_groups[2][row[2]]:
                        office += " "+x_sorted_annotations[j].description
                    office = office[1:]

            if len(bank_name)<8:
                continue

            csv_country.append([country_name, place, bank_name, office,0,0])
            if len(csv_country) == 1 and csv_country[0][1]== "":
                csv_country[0][1]="First row missing"
        
        # Curly braces detection
        dis_row = 60
        # Alternatively, calculate the distance between normal rows.
        try:
            dis_sorted = sorted(dis)
            dis_sorted = [dis_sorted[i] for i in range(len(dis_sorted)) if 0.25*len(dis_sorted)<i<0.75*len(dis_sorted)]
            dis_row = mean(dis_sorted)
        except:
            pass
        for j,row in enumerate(csv_country):
            if j == 0:

                if rows[0][0] != 0 and dis[j] >1.8*dis_row:
                    csv_country[j][4] = 1
                    #csv_country[j][5] = rows_height[0] 
                    csv_country[j][5] = int(dis[j]/dis_row)#+int(x_sorted_annotations[v5_groups[0][rows[0][0]][0]].bounding_poly.vertices[0].y/dis_row)
            elif j == len(csv_country)-1:
                if dis[j-1] >1.8*dis_row:
                    csv_country[j][4] = 1
                    csv_country[j][5] = int(dis[j-1]/dis_row)
            else:
                if dis[j-1] >1.8*dis_row and dis[j] >1.8*dis_row:
                    csv_country[j][4] = 1
                    csv_country[j][5] = int(dis[j-1]/dis_row)+int(dis[j]/dis_row)
                    if csv_country[j-1][4] == 1:
                        csv_country[j][5] -= int(0.5*dis[j-1]/dis_row)
                        csv_country[j-1][5] -= int(0.5*dis[j-1]/dis_row)

                

        csv_countries.append(csv_country)
    return csv_countries

def gen_excel_rows(csv_countries,YEAR,image_number,page):
    excel_rows = []
    for country in csv_countries:
        if len(country) == 0:
            continue
        for row in country:
            if len(excel_rows) == 0:
                excel_rows.append([YEAR,image_number,page]+row)
            else:
                excel_rows.append(["","",""]+row)
    
    return excel_rows

def clean_excel(excel_rows):
    for i,row in enumerate(excel_rows):
        num_c1 = 0
        for _ in row[4]:
            if _.isalpha():
                num_c1+=1

        if num_c1 < 4:
            excel_rows[i][4] = ""

        if excel_rows[i][4] == "":
            excel_rows[i][4] = excel_rows[i-1][4]


        num_c3 = 0
        for _ in row[6]:
            if _.isalpha():
                num_c3+=1

        if num_c3 < 4:
            excel_rows[i][6] = ""

        if excel_rows[i][6] == "":
            excel_rows[i][6] = excel_rows[i-1][6]

    return excel_rows


def detect_error(YEAR,image_number,page,num_v5,num_list,error_partition):
    if num_v5[1] < 25:
        type1_error = True
    else:
        type1_error = False
    if num_v5[0] < 25 or num_v5[1] < 25 or num_v5[2] < 25:
        type2_error = True
    else:
        type2_error = False   
    if abs(num_list[0]-num_v5[0]) > 20 or abs(num_list[1]-num_v5[1]) > 20 or abs(num_list[2]-num_v5[2]) > 20:
        type3_error = True
    else:
        type3_error = False
    check_error = [YEAR,image_number,page] + num_v5+num_list + [error_partition,type1_error,type2_error,type3_error] 
    
    return check_error 

def save_excel(excel_agg,check_agg,YEAR):
    df = pd.DataFrame(excel_agg)
    writer = pd.ExcelWriter("/zfs/projects/faculty/coppola-xu/data/output/"+YEAR+'output.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='sheet1', index=False,header=False)
    writer.save()

    df = pd.DataFrame(check_agg)
    writer = pd.ExcelWriter("/zfs/projects/faculty/coppola-xu/data/output/"+YEAR+'error.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='sheet1', index=False,header=False)
    writer.save()