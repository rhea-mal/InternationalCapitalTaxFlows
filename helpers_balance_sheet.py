import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from helpers_location_v5 import draw_anno_bounding
from copy import deepcopy
import random
from statistics import median

def show_plot(img,size = (15,15)):
    plt.figure(figsize=size)
    plt.imshow(img)
    plt.show()

def get_number(filename):
    i, j = filename.index("("), filename.index(")")
    image_number = int(filename[i+1:j])
    if filename.endswith('1L.jpg') or filename.endswith('1L.tif'):
        page = "1L"
    else:
        page = "2R"
    return image_number, page

def gen_list(x_sorted_annotations,img, show = False):
    img_0 = deepcopy(img)
    number_list = []
    good_words = ["LIA","ASSET","Total"]
    liability_list = []
    asset_list = []
    total_list = []

    for i,anno in enumerate(x_sorted_annotations):
        des = anno.description
        num = 0
        for letter in des:
            if letter.isdigit(): num+=1
        if num/len(des)>0.5:
            if len(des)> 3:
                number_list.append(i)
        if "LIA" in des:
            liability_list.append(i)
        if "ASSET" in des:
            asset_list.append(i)
        if "Total" in des:
            total_list.append(i)
        
        if show:
            for word in good_words:
                if word in des:
                    color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
                    pts = np.array([[[vertex.x, vertex.y] for vertex in anno.bounding_poly.vertices]], np.int32)
                    cv2.polylines(img_0, [pts], True, color, 3)     

    if show:
        color = (0,255,0)
        for i,anno in enumerate(x_sorted_annotations):
                if i in number_list:
                        pts = np.array([[[vertex.x, vertex.y] for vertex in anno.bounding_poly.vertices]], np.int32)
                        cv2.polylines(img_0, [pts], True, color, 3)
        show_plot(img_0) 

    return number_list,liability_list,asset_list,total_list

def grouping_format(annotations,liability_list,asset_list,total_list):
    lia_ast = []
    for lia in liability_list:
        for ast in asset_list:
            if abs(annotations[lia].bounding_poly.vertices[0].y-annotations[ast].bounding_poly.vertices[0].y)<200:
                lia_ast.append((lia,ast))
    lia_ast= sorted(lia_ast,key = lambda x:annotations[x[0]].bounding_poly.vertices[0].y)
    
    ttl1_ttl2 = []
    for ttl_1 in total_list:
        for ttl_2 in total_list:
            if abs(annotations[ttl_1].bounding_poly.vertices[0].y-annotations[ttl_2].bounding_poly.vertices[0].y)<200:
                if annotations[ttl_2].bounding_poly.vertices[0].x-annotations[ttl_1].bounding_poly.vertices[0].x > 500:
                    ttl1_ttl2.append((ttl_1,ttl_2))
    ttl1_ttl2= sorted(ttl1_ttl2,key = lambda x:annotations[x[0]].bounding_poly.vertices[0].y)

    lia_ast_ttl1_ttl2 = list(zip(lia_ast,ttl1_ttl2))
    lia_ast_ttl1_ttl2 = [(_[0][0],_[0][1],_[1][0],_[1][1]) for _ in lia_ast_ttl1_ttl2]

    return lia_ast_ttl1_ttl2

def bs_grouping_element(img_shape,x_sorted_annotations, para_1 = [30,100,30],show = False):
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
    
    if show: show_plot(img_group_column)

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
    
    return groups[1:]

def grouping_digits(number_list,x_sorted_annotations):
    number_list = sorted(number_list, key= lambda x: x_sorted_annotations[x].bounding_poly.vertices[1].y)
    added = set()
    groups = []
    for number_1 in number_list:
        des = x_sorted_annotations[number_1].description
        if len(des) < 5 or "-" in des:
            continue
        if number_1 in added:
            continue
        group = [number_1]
        added.add(number_1)
        x1_up = x_sorted_annotations[number_1].bounding_poly.vertices[1].x
        y1_up = x_sorted_annotations[number_1].bounding_poly.vertices[1].y
        x1_down = x_sorted_annotations[number_1].bounding_poly.vertices[2].x
        y1_down = x_sorted_annotations[number_1].bounding_poly.vertices[2].y
        for number_2 in number_list:
            des = x_sorted_annotations[number_2].description
            if len(des) < 5 :
                continue
            if number_2 in added:
                continue
            x2_up = x_sorted_annotations[number_2].bounding_poly.vertices[1].x
            y2_up = x_sorted_annotations[number_2].bounding_poly.vertices[1].y
            x2_down = x_sorted_annotations[number_2].bounding_poly.vertices[2].x
            y2_down = x_sorted_annotations[number_2].bounding_poly.vertices[2].y
            if -5 < y2_up - y1_down < 130 and abs(x2_up-x1_down) < 150:
                group.append(number_2)
                added.add(number_2)
                x1_down = x2_down
                y1_down = y2_down
        groups.append(group)
        groups = [group for group in groups if len(group)>1]
        groups = sorted(groups,key= lambda x: x_sorted_annotations[x[0]].bounding_poly.vertices[0].x)
    return groups

def gen_bs(x_sorted_annotations,lia_ast_ttl1_ttl2,bs_groups,digit_groups):
    agg = []
    accounts = []
    lia_ast_ttl1_ttl2 = sorted(lia_ast_ttl1_ttl2, key= lambda x: x_sorted_annotations[x[0]].bounding_poly.vertices[0].y)
    for i,_ in enumerate(lia_ast_ttl1_ttl2):
        lia,ast,ttl1,ttl2 = _
        x_lia = x_sorted_annotations[lia].bounding_poly.vertices[0].x
        y_lia = x_sorted_annotations[lia].bounding_poly.vertices[0].y
        x_ast = x_sorted_annotations[ast].bounding_poly.vertices[0].x
        y_ast = x_sorted_annotations[ast].bounding_poly.vertices[0].y
        x_ttl1 = x_sorted_annotations[ttl1].bounding_poly.vertices[0].x
        y_ttl1 = x_sorted_annotations[ttl1].bounding_poly.vertices[0].y
        x_ttl2 = x_sorted_annotations[ttl2].bounding_poly.vertices[0].x
        y_ttl2 = x_sorted_annotations[ttl2].bounding_poly.vertices[0].y    
        
        items_lia = []
        items_ast = []
        bs_groups = sorted(bs_groups, key= lambda x:x_sorted_annotations[x[0]].bounding_poly.vertices[0].y)
        for group in bs_groups:
            x1 = x_sorted_annotations[group[0]].bounding_poly.vertices[0].x
            y1 = x_sorted_annotations[group[0]].bounding_poly.vertices[0].y
            if x_lia < x1 < x_ttl1 and y_lia < y1 < y_ttl1:
                items_lia.append(group)
            if x_ast < x1 < x_ttl2 and y_ast < y1 < y_ttl2:
                items_ast.append(group) 
        #print(x_lia,x_ttl1,y_lia,y_ttl1,x_ast,x_ttl2,y_ast,y_ttl2,items_lia,items_ast)
            
        poss_digit_lia = []
        poss_digit_ast = []
        for group in digit_groups:
            x_dig = x_sorted_annotations[group[0]].bounding_poly.vertices[0].x
            y_dig = x_sorted_annotations[group[0]].bounding_poly.vertices[0].y
            if x_lia < x_dig < x_ast and abs(y_lia-y_dig) < 250:
                poss_digit_lia.append(group)
            if x_dig > x_ast and abs(y_ast-y_dig) < 200:
                poss_digit_ast.append(group)
        
        digit_l_lia = poss_digit_lia[0]
        digit_r_lia = poss_digit_lia[1]
        digit_l_ast = poss_digit_ast[0]
        digit_r_ast = poss_digit_ast[1]
        for account in items_lia:
            accounts.append(account)
        for account in items_ast:
            accounts.append(account)

        agg.append((items_lia,digit_l_lia,digit_r_lia,items_ast,digit_l_ast,digit_r_ast))

    return agg, accounts

def detect_bank_name(x_sorted_annotations,liability_list,asset_list,img_shape):
    capital_list = []
    for i,anno in enumerate(x_sorted_annotations):
        if i in liability_list or i in asset_list:
            continue
        if anno.description.isupper() or "&" in anno.description:
            capital_list.append(i)

    added = set()
    bank_names = []
    for i in capital_list:
        if i in added:
            continue
        added.add(i)
        bank_name = [i]
        x = x_sorted_annotations[i].bounding_poly.vertices[1].x
        y = x_sorted_annotations[i].bounding_poly.vertices[1].y
        for j in capital_list:
            if j in added:
                continue
            x1 = x_sorted_annotations[j].bounding_poly.vertices[0].x
            y1 = x_sorted_annotations[j].bounding_poly.vertices[0].y        
            x2 = x_sorted_annotations[j].bounding_poly.vertices[1].x
            y2 = x_sorted_annotations[j].bounding_poly.vertices[1].y
            if 0<x1-x<100 and abs(y1-y) <100:
                x = x2
                y = y2
                bank_name.append(j)
                added.add(j)
        bank_names.append(bank_name)

    '''
    length_list = []
    cleaned_bank_names = []
    for anno in x_sorted_annotations:
        length_list.append(anno.bounding_poly.vertices[3].y-anno.bounding_poly.vertices[0].y)
    avg_length = median(length_list)
    for bank_name in bank_names:
        length = x_sorted_annotations[bank_name[0]].bounding_poly.vertices[3].y - x_sorted_annotations[bank_name[0]].bounding_poly.vertices[0].y
        if length > 1.2*avg_length:
            cleaned_bank_names.append(bank_name)'''

    bank_names = [name for name in bank_names if len(name)>1 and x_sorted_annotations[name[0]].description != "&"]
    real_bank_names = []
    for group in bank_names:
        des =""
        for i in group:
            des+= x_sorted_annotations[i].description
        #if "." in des:
            #continue
        if len(des) < 4:
            continue
        if "INTERNATIONAL" in des or "ALMANAC" in des or "OTHER" in des:
            continue
        if x_sorted_annotations[group[-1]].bounding_poly.vertices[1].x > img_shape[1]*0.66:
            continue
        real_bank_names.append(group)
    real_bank_names = sorted(real_bank_names,key= lambda x: x_sorted_annotations[x[0]].bounding_poly.vertices[0].y)

    real_bank_names = [name for name in real_bank_names if name[0]<len(x_sorted_annotations)/10]

    return real_bank_names

def gen_rows(bank_names,x_sorted_annotations,agg,liability_list):
    agg_all = []
    idx_banks = []
    for i in range(len(agg)):
        #idx_bank_name = bank_names[i]
        # liability_list
        idx_bank = []
        elements = agg[i]
        liability_list = sorted(liability_list,key= lambda x: x_sorted_annotations[x].bounding_poly.vertices[0].y)
        y_lia = x_sorted_annotations[liability_list[i]].bounding_poly.vertices[0].y
        
        for i in elements:
            for j in i:
                try:
                    idx_bank+=j
                except:
                    idx_bank.append(j)

        dis_store = y_lia - x_sorted_annotations[bank_names[0][0]].bounding_poly.vertices[0].y
        idx_store = 800
        for idx in bank_names:
            dis = y_lia - x_sorted_annotations[idx[0]].bounding_poly.vertices[0].y
            if dis > 200 and dis <= dis_store:
                dis_store = dis
                idx_store = idx
        if idx_store == 800:
            for idx in bank_names:
                dis = y_lia - x_sorted_annotations[idx[0]].bounding_poly.vertices[0].y
                if dis <= dis_store:
                    dis_store = dis
                    idx_store = idx            

        des =""
        for k in idx_store:
            des+= x_sorted_annotations[k].description + " "
            idx_bank.append(k)
        bank_name = des
        row_0 = [bank_name,"","","","",""]
        row_1 = ["Liabilities","","","Assets","",""]
        agg_all.append(row_0)
        agg_all.append(row_1)

        # Need imporvement
        num_row = [0,0]
        act_c1,num_row[0] = elements[1][:-1],len(elements[1][:-1])
        act_c2 = elements[2][:-1]
        act_c3,num_row[1] = elements[4][:-1],len(elements[4][:-1])
        act_c4 = elements[5][:-1]
        num_rows = max(num_row)
        
        for j in range(num_rows):
            des_0 =""
            try:
                for k in elements[0][j]:
                    des_0 += x_sorted_annotations[k].description + " "
            except:
                pass       
            des_1 =""
            try:
                for k in elements[3][j]:
                    des_1 += x_sorted_annotations[k].description + " "
            except:
                pass 
            dig_1,dig_2,dig_3,dig_4 = "","","",""
            try:
                dig_1 = x_sorted_annotations[act_c1[j]].description
            except:
                pass
            try:
                dig_2 = x_sorted_annotations[act_c2[j]].description
            except:
                pass
            try:
                dig_3 = x_sorted_annotations[act_c3[j]].description
            except:
                pass
            try:
                dig_4 = x_sorted_annotations[act_c4[j]].description           
            except:
                pass
            agg_all.append([des_0,dig_1,dig_2,des_1,dig_3,dig_4])
        dig_1 =x_sorted_annotations[elements[1][-1]].description
        dig_2 =x_sorted_annotations[elements[2][-1]].description
        dig_3 =x_sorted_annotations[elements[4][-1]].description
        dig_4 =x_sorted_annotations[elements[5][-1]].description
        agg_all.append(["Total",dig_1,dig_2,"Total",dig_3,dig_4])
        idx_banks.append(idx_bank)
    return agg_all,idx_banks

def save_excel_bs(rows,excel_name):
    df = pd.DataFrame(rows)
    writer = pd.ExcelWriter("/zfs/projects/faculty/coppola-xu/analysis/"+excel_name, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='sheet1', index=False,header=False)
    writer.save()

def createGroups_bs(sorted_annotations):
    added = set()
    ave_height = np.average([anno.bounding_poly.vertices[3].y - anno.bounding_poly.vertices[0].y for anno in sorted_annotations])

    groups = []
    for i, annotation in enumerate(sorted_annotations):
        #if annotation.bounding_poly.vertices[0].x< second - 20 or annotation.bounding_poly.vertices[1].x> third:
            #continue
        if i in added:
            continue
        xmin = np.min([v.x for v in annotation.bounding_poly.vertices])
        xmax = np.max([v.x for v in annotation.bounding_poly.vertices])
        ymin = np.min([v.y for v in annotation.bounding_poly.vertices])
        ymax = np.max([v.y for v in annotation.bounding_poly.vertices])

        if ymax - ymin > ave_height*1.5:
            continue
        # if ymax - ymin < 12:
        #     continue
        group = [i]

        added.add(i)
        just_xmin, just_xmax, just_ymin, just_ymax = xmin,xmax,ymin,ymax
        for j, p_ann in enumerate(sorted_annotations):
            if j in added:
                continue
            p_xmin = np.min([v.x for v in p_ann.bounding_poly.vertices])
            p_xmax = np.max([v.x for v in p_ann.bounding_poly.vertices])
            p_ymin = np.min([v.y for v in p_ann.bounding_poly.vertices])
            p_ymax = np.max([v.y for v in p_ann.bounding_poly.vertices])
            if p_ymax < just_ymin or p_ymin > just_ymax:
                continue
            # if p_ymax - p_ymin < 12:
            #     continue
            line_overlap = np.min([p_ymax - just_ymin, just_ymax - p_ymin]) / np.max([p_ymax-p_ymin, just_ymax-just_ymin])
            if p_ymin >= just_ymax or p_ymax <= just_ymin:
                line_overlap = 0
            elif p_ymin >= just_ymin and p_ymax <= just_ymax:
                line_overlap = 1
            elif p_ymin <= just_ymin and p_ymax >= just_ymax:
                line_overlap = 1
            elif p_ymin >= just_ymin:
                line_overlap = ( just_ymax - p_ymin) / (just_ymax-just_ymin)
            elif p_ymin <= just_ymin:
                line_overlap = (p_ymax - just_ymin) / (just_ymax-just_ymin)
            #if line_overlap < 0.6:
            if line_overlap < 0.6:
                continue
            group.append(j)
            added.add(j)
            
            #just_xmin, just_xmax, just_ymin, just_ymax = p_xmin, p_xmax, p_ymin, p_ymax
        if len(group) == 1:
            if len(sorted_annotations[group[0]].description) < 4:
                continue
        group = sorted(group, key=lambda x: sorted_annotations[x].bounding_poly.vertices[0].x)
        if (not sorted_annotations[group[0]].description.isalnum) or sorted_annotations[group[0]].description.islower(): group = group[1:]
        if len(group) != 0:
            groups.append(group)
    groups = sorted(groups, key=lambda x: sorted_annotations[x[0]].bounding_poly.vertices[0].y)
    return groups

def show_groups(groups_groups,img,x_sorted_annotations,thickness = 6,colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]):
    img_0 = deepcopy(img)
    for i,groups in enumerate(groups_groups):
        color = colors[i]
        for group in groups:
            #color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
            for i in group:
                pts = np.array([[[vertex.x, vertex.y] for vertex in x_sorted_annotations[i].bounding_poly.vertices]], np.int32)
                cv2.polylines(img_0, [pts], True, color, thickness)
    show_plot(img_0)