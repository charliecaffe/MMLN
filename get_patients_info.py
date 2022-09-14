# 从EMR中提取阳性症状，详见extract_positive_symptoms函数
# -*- coding: utf-8 -*-
''' extract useful info from EMR & de-identified '''
from genericpath import exists
from logging import disable
import os
import json
import jieba.posseg as psg
import jieba
import codecs
import csv
import re


symptom_path = 'symptom.txt'
symptom_wds = [i.strip() for i in open(symptom_path, encoding='UTF8') if i.strip()]
selected_emrs = []
os.chdir('EMR_utf8')

class Text:
    def __init__(self):    
        self.ry = ''
        self.bc = ''
        self.cf = ''
        self.tg = ''
        self.cy = ''    
        self.strlist = {}
        self.dir = ''
        self.filenames = []
        # self.Json_list = self.solve()

    # 入院记录
    def ry_record(self):
        file_path = os.path.join(self.dir, self.ry).replace("\\", "/")
        file = open(file_path, encoding='utf8')
        file = file.read()

        self.strlist['ID'] = self.dir
        # 基本信息
        pos = {}
        pos['性别'] = file.find("性别")
        pos['年龄'] = file.find("年龄")
        pos['职业'] = file.find("职业")
        pos['入院日期'] = file.find('入院日期')
        pos['出生地'] = file.find('出生地')
        pos['民族'] = file.find('民族')

        for key in pos:
            if pos[key] != -1 and (file[pos[key] + len(key)] == ':' or file[pos[key] + len(key)] == '：'):
                enter = file.find('\n', pos[key])
                string = file[pos[key] + len(key) : enter]
                if key == '年龄':
                    string = re.search('[0-9]+', string).group(0)
                    # string = string.replace('Y', '')
                    # string = string.replace('岁', '')
                else:
                    string = re.sub('：|:', '', string)
                    # string = string.replace('：', '')
                    # string = string.replace(':', '')
                self.strlist[key] = string
            else:
                self.strlist[key.replace(' ', '')] = "无"

        if file.find('月经史') != -1:
            name = ['主  诉', '现病史', '既往史', '个人史', '婚育史', '月经史', '家族史', '\n']
        else:
            name = ['主  诉', '现病史', '既往史', '个人史', '婚育史', '家族史', '\n']

        for j in range(len(name) - 1):
            pos = file.find(name[j])
            if pos == -1:
                pos = file.find(name[j].replace(' ', ''))
            pos_next = file.find(name[j + 1])
            if pos_next == -1:
                pos_next = file.find(name[j + 1].replace(' ', ''))
            string = "无"

            if pos != -1 and name[j] != '\n':
                if name[j] == '家族史':
                    pos_next = file.find(name[j + 1], pos)
                    string = file[pos + len(name[j]):pos_next]
                else:
                    string = file[pos + len(name[j]):pos_next]
                string = string.replace(':', '')
                string = string.replace('：', '')
                string = string.replace('\r', '')
                string = string.replace('\n', '')
                string = string.replace(' ', '')
                string = string.replace('（一）', '')
                string = string.replace('（二）', '')
                string = string.replace('（三）', '')
                self.strlist[name[j].replace(' ', '')] = string
                if name[j] == '婚育史':
                    self.strlist['月经史'] = '无'
            if pos == -1 and name[j] != '\n':
                self.strlist[name[j].replace(' ', '')] = '无'
    
    # 体格检查
    def tg_record(self):
        file_path = os.path.join(self.dir, self.tg).replace("\\", "/")
        file = open(file_path, encoding='utf8')
        file = file.read()
        # 体格检查
        name = ['一 般 情 况', '专 科 情 况', '辅 助 检 查', '刷 新 诊 断']
        for j in range(len(name) - 1):
            pos = file.find(name[j])
            if pos == -1:
                pos = file.find(name[j].replace(' ', ''))
            pos_next = file.find(name[j + 1])
            if pos_next == -1:
                pos_next = file.find(name[j + 1].replace(' ', ''))
            string = file[pos + len(name[j]):pos_next]
            string = string.replace(':', '')
            string = string.replace('：', '')
            string = string.replace('\r', '')
            string = string.replace('\n', '')
            string = string.replace(' ', '')
            string = string.replace('（一）', '')
            string = string.replace('（二）', '')
            string = string.replace('（三）', '')
            self.strlist[name[j].replace(' ', '')] = string
            if pos == -1:
                self.strlist[name[j].replace(' ', '')] = '无'

    # 首次查房记录
    def cf_record(self):
        file_path = os.path.join(".", self.dir).replace("\\", "/")
        file_path = os.path.join(file_path, self.cf).replace("\\", "/")
        file = open(file_path, encoding='utf8')
        file = file.read()

        # 病情
        pos1 = file.find('查体')
        if pos1 != -1:
            string = file[pos1 + len('查体'):len(file)]
            string = string.replace(':', '')
            string = string.replace('：', '')
            string = string.replace('\r', '')
            string = string.replace('\n', '')
            string = string.replace(' ', '')
            self.strlist['查体'] = string
        else:
            self.strlist['查体'] = '无'
    
    # 首次病程记录
    def bc_record(self):
        file_path = os.path.join(self.dir, self.bc).replace("\\", "/")
        file = open(file_path, encoding='utf8')
        file = file.read()

        # 病情
        name = ['病例特点', '诊断、诊断依据及鉴别诊断', '诊疗计划']
        for j in range(len(name)):
            pos = file.find(name[j])
            if j != len(name) - 1:
                pos_next = file.find(name[j + 1])
            else:
                pos_next = len(file)
            string = "无"
            if pos != -1:
                if name[j] != '诊疗计划':
                    string = file[pos + len(name[j]):pos_next]
                else:
                    string = file[pos + len(name[j]):pos_next]
            string = string.replace(':', '')
            string = string.replace('：', '')
            string = string.replace('\r', '')
            string = string.replace('\n', '')
            string = string.replace(' ', '')
            string = string.replace('（一）', '')
            string = string.replace('（二）', '')
            string = string.replace('（三）', '')
            self.strlist[name[j]] = string

    # 出院记录
    def cy_record(self): 
        file_path = os.path.join(".", self.dir).replace("\\", "/")
        file_path = os.path.join(file_path, self.cy).replace("\\", "/")
        file = open(file_path, encoding='utf8')
        file = file.read()

        names = ['入院诊断', '出院诊断']
        for name in names:
            string = "无"
            pos = file.find(name)
            if pos != -1:
                pos_next = file.find('\n', pos)
                string = file[pos + len(name) + 1 : pos_next]
            # string = string.replace(':', '')
            # string = string.replace('：', '')
            # string = string.replace('\r', '')
            # string = string.replace('\n', '')
            # string = string.replace(' ', '')
            # string = string.replace('（一）', '')
            # string = string.replace('（二）', '')
            # string = string.replace('（三）', '')
            self.strlist[name] = string

    def main_diagnosis(self):
        cy_diagnosis = self.strlist['出院诊断']
        if cy_diagnosis == '' or cy_diagnosis == '无':
            return False
        # pos_begin = cy_diagnosis.find('初步诊断')
        # pos_end = cy_diagnosis.find('诊断依据')
        # cy_diagnosis = cy_diagnosis[pos_begin + len('初步诊断'):pos_end]

        # cy_diagnosis = cy_diagnosis.replace('1、', ' ')
        # cy_diagnosis = cy_diagnosis.replace('2、', ' ')
        # cy_diagnosis = cy_diagnosis.replace('3、', ' ')
        # cy_diagnosis = cy_diagnosis.replace('①', ' ')
        # cy_diagnosis = cy_diagnosis.replace('②、', ' ')
        # cy_diagnosis = cy_diagnosis.replace('③、', ' ')
        # cy_diagnosis = cy_diagnosis.replace('1.', ' ')
        # cy_diagnosis = cy_diagnosis.replace('2.', ' ')
        # cy_diagnosis = cy_diagnosis.replace('3.', ' ')
        cy_diagnosis = re.sub('1、|2、|3、|①|②|③|1.|2.|3.', ' ', cy_diagnosis)

        main_diagnosis = cy_diagnosis.split(' ')
        while '' in main_diagnosis:
            main_diagnosis.remove('')
        if len(main_diagnosis) == 0:
            return False
        else:
            top_disease = main_diagnosis[0]
        # 第一诊断为肺炎, 诊断总数<4, 年龄<40, 入院次数<4
        visit_id = self.strlist['ID'].split('_')[-1]
        if '肺炎' in top_disease and len(main_diagnosis) < 4 and int(visit_id) < 4 and int(self.strlist['年龄']) < 40:
            selected_emrs.append(self.strlist['ID'])
        self.strlist['首要诊断'] = top_disease

    def output_disease_to_symptom(self):
        path = "../disease_to_symptom.txt"
        file = open(path, "w", encoding='utf8')
        file.seek(0, 0)
        file.truncate()
        # 聚类
        diseases = {}
        for json in self.Json_list:
            disease_name = json['疾病']
            # 加入新的疾病
            if disease_name not in diseases:
                diseases[disease_name] = {}
                diseases[disease_name]['患病人数'] = 0
            # 疾病人数++
            diseases[disease_name]['患病人数'] = diseases[disease_name]['患病人数'] + 1

            exist_symptom = diseases[disease_name]
            
            symptoms = json['阳性症状']
            for symptom in symptoms:
                if symptom == disease_name:
                    continue
                # 创建新的症状
                if symptom not in exist_symptom:
                    exist_symptom[symptom] = 0
                # 该疾病有该症状的概率++
                exist_symptom[symptom] = exist_symptom[symptom] + 1
                # print(disease_name," ",symptom," ","患病人数",exist_symptom['患病人数'])

        for disease_name, info in diseases.items():
            # file.write('疾病名称:'+disease_name+',总患病人数:'+str(info['患病人数'])+'\n')
            for symptom, nums in info.items():
                if symptom != '患病人数':
                    file.write(
                        disease_name + '\tdisease_to_symptom\t' + symptom + '\t' + str(nums / info['患病人数']) + '\n')

    def output_disease_to_symptom_xucy(self):
        # 读取json每一行的阳性症状
        cases = json.load('J18_9.json')

    def output_disease_to_check(self):
        path = "../disease_to_check.txt"
        file = open(path, "w", encoding='utf8')
        file.seek(0, 0)
        file.truncate()
        # 聚类
        diseases = {}
        for json in self.Json_list:
            disease_name = json['疾病']
            # 加入新的疾病
            if disease_name not in diseases:
                diseases[disease_name] = {}
                diseases[disease_name]['患病人数'] = 0
            # 疾病人数++
            diseases[disease_name]['患病人数'] = diseases[disease_name]['患病人数'] + 1

            exist_check = diseases[disease_name]
            checks = json['指标']
            for check, level in checks.items():
                # 创建新的症状
                if check not in exist_check:
                    exist_check[check] = {'指标': level, '人数': 0}
                exist_check[check]['人数'] = exist_check[check]['人数'] + 1

        for disease_name, info in diseases.items():
            for check_name, value in info.items():
                if check_name != '患病人数':
                    file.write(disease_name + '\tdisease_to_check\t' + check_name + '\t' + value['指标'] + '\t' + str(
                        value['人数'] / info['患病人数']) + '\n')

    def _extract_pos_syms(self, paragraph):
        paragraph = re.sub("[ ]+", "", paragraph)
        paragraph = re.sub("[.|;|；]+", "。", paragraph)
        sep_list = jieba.cut(paragraph, cut_all=False)
        sep = list(sep_list)
        symptoms = [word for word in sep if word in symptom_wds]
        # 先分句
        sentences = re.split('。', paragraph)
        sentences = [item for item in sentences if item != ""]
        neg_syms = []
        pos_syms = []
        for sentence in sentences:
            neg_phrases = re.findall("无[^(明显向。无)]+", sentence)
            for neg_phrase in neg_phrases:
                neg_syms += [word for word in symptoms if word in neg_phrase]

            pos_phrases = re.findall("有[^(。,，无)]*", sentence)
            for pos_phrase in pos_phrases:
                pos_syms += [word for word in symptoms if word in pos_phrase]

        pos_syms = (set(symptoms) - set(neg_syms)) | set(pos_syms)
        symptoms = set(symptoms)
        return pos_syms, symptoms

    def extract_positive_symptoms(self):
        disease_history = self.strlist['现病史']
        main_complaint = self.strlist['主诉']
        pos_syms, symptoms = self._extract_pos_syms(disease_history)
        pos_syms_main, symptoms_main = self._extract_pos_syms(main_complaint)
        self.strlist['阳性症状'] = list(pos_syms | pos_syms_main)

    def get_indicators(self):
        check_body = self.strlist['查体']
        indicators = []
        check_body = re.split('[，。；,]+', check_body)
        # print(check_body)
        for word in check_body:
            if word.find('↑') == -1 and word.find('↓') == -1:
                continue
            word=re.sub(".*\d+-\d+-\d+","",word)
            pos = re.search(r'.*[\u4E00-\u9FFF]+[）)]*', word)
            if not pos:
                pos = re.search(r'\w+[^(]+', word)
            if pos:
                indicator = pos.group()
                # print(indicator)
                if word.find('↑') != -1:
                    indicators.append(indicator + ':' + '过高')
                    # indicators[indicator] = '过高'
                elif word.find('↓') != -1:
                    # indicators[indicator] = '过低'
                    indicators.append(indicator + ':' + '过低')
        return indicators

    def solve(self):
        Json_list = list()
        dirs = os.listdir(".")
        # for dir in ['N528349_1', 'N526072_1']:
        for dir in dirs:
            self.dir = dir
            self.ry = ''
            self.bc = ''
            self.cf = ''
            self.tg = ''
            self.cy = ''
            self.strlist = {}

            if dir == '.idea' or dir == 'dict':
                continue
            self.filenames = [file for root, dirs, files in os.walk(os.path.join(".", dir)) for file in files]
            if len(self.filenames) == 0 or (not os.path.isdir(os.path.join(".", dir))):
                continue
                        
            for filename in self.filenames:
                file_path = os.path.join(".", dir).replace("\\", "/")
                file_path = os.path.join(file_path, filename).replace("\\", "/")
                file = open(file_path, encoding='utf8')
                file = file.read()
                if file.find('基本信息') != -1 and (file.find('出生地') != -1 or file.find('婚姻状况') != -1):
                    self.ry = filename

                if file.find('体格检查') != -1 or file.find('体 格 检 查') != -1:
                    self.tg = filename

                if '术后首次病程记录' not in filename and (
                        '首次病程记录' in filename or file.find('首次病程记录') != -1) and self.bc == '':
                    self.bc = filename

                if ('首次查房记录' in filename or file.find('首次查房记录') != -1) and self.cf == '':
                    self.cf = filename

                # 病情危重而去世的人只有死亡记录，因此没有出院记录，这里自动筛掉了
                if '出院记录' in filename and file.find('出院记录') != -1:
                    self.cy = filename

            if self.ry == '':
                print('无入院记录：', self.dir)
                continue

            self.ry_record()

            if self.cy != '':
                self.cy_record()
            else:
                name = ['入院诊断', '出院诊断']
                for j in name:
                    self.strlist[j] = '无' 

            # 如果提取不了主要诊断，就跳过这个病例
            if self.main_diagnosis() == False:
                print('无主要诊断：', self.strlist['ID'])
                continue

            if self.tg != '':
                self.tg_record()
            else:
                name = ['一般情况', '专科情况', '辅助检查']
                for j in name:
                    self.strlist[j] = '无'

            if self.bc != '':
                self.bc_record()
            else:
                name = ['病例特点', '诊断、诊断依据及鉴别诊断', '诊疗计划']
                for j in name:
                    self.strlist[j] = '无'

            if self.cf != '':
                self.cf_record()
            else:
                for filename in self.filenames:
                    if '查房记录' in filename:
                        self.cf = filename
                        self.cf_record()
                        break

            if self.cf == '':
                self.strlist['查体'] = '无'
            
            self.strlist['指标'] = self.get_indicators()

            self.extract_positive_symptoms()
            # Json = json.dumps(self.strlist, ensure_ascii=False)
            Json_list.append(self.strlist)
        return Json_list


if __name__ == "__main__":
    text = Text()
    json_list = text.solve()

    # output the selected emrs
    with open('../pvid_selected.txt', 'w', encoding='utf8') as f:
        for i in selected_emrs:
            f.write(i)
            f.write('\n')

    # output the patient json
    with open('../J18_9.json', "w", encoding='utf8') as f:
        for i in json_list:
            json.dump(i, f, ensure_ascii=False)
            f.write('\n')
    # text.output_disease_to_symptom()
    # text.output_disease_to_check()
