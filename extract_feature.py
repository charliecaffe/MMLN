'''从EMR中提取特征：阳性症状、白细胞、湿性啰音、肺部炎症'''
import json
import csv
import re
import os

codes = ['A15', 'A16', 'C34', 'I26', 'J18']
# codes = ['A15']
for code in codes:
    # id_input = os.path.join(code, code+'_EMR_CXR_ID_1.txt')
    # emr_input = os.path.join(code, code+'_EMR_selected_1.json')
    # output = os.path.join('output', code+'_features_1.json')

    emr_input = os.path.join('MMLN', 'EMR_selected_{}.json'.format(code))
    output = os.path.join('MMLN', 'features_{}.json'.format(code))

    fields = ['查体', '辅助检查', '病例特点', '诊断、诊断依据及鉴别诊断', '一般情况', '专科情况']
    items = ['白细胞', '湿性啰音', '肺部炎症']
    patterns = ['白细胞[(|（]*WBC[)|）]*[^×/]+×10\^9/L', '[未|可]*闻及[^0-9，,]*湿性啰音', '[上下左右中双两侧]*肺[^，。；;,.、?？并]*?(炎症|感染|炎){1,2}?']
    json_list = []


    with open(emr_input, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            case = json.loads(line)
            id = case['ID']
            features = {}
            features['ID'] = case['ID']
            for item in items:
                features[item] = []

            for field in fields:
                if field in case.keys():
                    for idx, item in enumerate(items):
                        temp = re.search(patterns[idx], case[field])
                        if temp != None:
                            if temp.group(0) not in features[item]:
                                features[item].append(temp.group(0))
            features['阳性症状'] = case['阳性症状']
            json_list.append(json.dumps(features, ensure_ascii=False))

    with open(output, 'w', encoding='utf8') as g:
        for i in json_list:
            g.write(i + '\n')



# '''提取特征：阳性症状、白细胞、湿性啰音、肺部炎症'''
# import json
# import csv
# import re
# import os

# codes = ['A15', 'A16', 'C34', 'I26', 'J18']
# # codes = ['A15']
# for code in codes:
#     # id_input = os.path.join(code, code+'_EMR_CXR_ID_1.txt')
#     # emr_input = os.path.join(code, code+'_EMR_selected_1.json')
#     # output = os.path.join('output', code+'_features_1.json')
#     id_input = os.path.join('output', 'EMR_CXR_ID_{}.txt'.format(code))
#     emr_input = os.path.join(code, code+'_EMR.json')
#     output = os.path.join('output', 'features_{}.json'.format(code))

#     fields = ['查体', '辅助检查', '病例特点', '诊断、诊断依据及鉴别诊断', '一般情况', '专科情况']
#     items = ['白细胞', '湿性啰音', '肺部炎症']
#     patterns = ['白细胞[(|（]*WBC[)|）]*[^×/]+×10\^9/L', '[未|可]*闻及[^0-9，,]*湿性啰音', '[上下左右中双两侧]*肺[^，。；;,.、?？并]*?(炎症|感染|炎){1,2}?']
#     json_list = []

#     with open(id_input, 'r') as f:
#         id_list = f.readlines()
#     id_list = [id[:-1] for id in id_list]
#     # print(id_list)

#     with open(emr_input, 'r', encoding='utf8') as f:
#         lines = f.readlines()
#         for line in lines:
#             case = json.loads(line)
#             id = case['ID']
#             if id in id_list:
#                 features = {}
#                 features['ID'] = case['ID']
#                 for item in items:
#                     features[item] = []

#                 for field in fields:
#                     if field in case.keys():
#                         for idx, item in enumerate(items):
#                             temp = re.search(patterns[idx], case[field])
#                             if temp != None:
#                                 if temp.group(0) not in features[item]:
#                                     features[item].append(temp.group(0))
#                 features['阳性症状'] = case['阳性症状']
#                 json_list.append(json.dumps(features, ensure_ascii=False))

#     with open(output, 'w', encoding='utf8') as g:
#         for i in json_list:
#             g.write(i + '\n')