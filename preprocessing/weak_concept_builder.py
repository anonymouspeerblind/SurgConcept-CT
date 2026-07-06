import os, json
import pandas as pd
from tqdm import tqdm

class Concept_builder():
    def __init__(self, split):
        self.split       = split
        self.df          = pd.read_csv("Raw clinical data for train/val/test file")
        self.concept_dic = dict()
    def extracting_raw_val(self):
        self.df_dic = self.df.set_index("CASE_ID").to_dict(orient="index")
    def age_burden(self):
        if self.df_dic_case['Age'] > 70:
            return 1.0
        elif self.df_dic_case['Age'] <= 70 and self.df_dic_case['Age'] >= 65:
            return 0.5
        else:
            return 0.0
    def low_bmi(self):
        if self.df_dic_case['BMI'] < 18.5:
            return 1.0
        else:
            return 0.0
    def obesity(self):
        if self.df_dic_case['BMI'] >= 30:
            return 1.0
        else:
            return 0.0
    def low_bmi_high_age(self):
        if self.df_dic_case['BMI'] < 18.5 and self.df_dic_case['Age'] >= 65:
            return 1.0
        else:
            return 0.0
    def airflow_reserve(self):
        if self.df_dic_case['FEV1 Predicted'] < 40:
            return 1.0
        elif self.df_dic_case['FEV1 Predicted'] >= 40 and self.df_dic_case['FEV1 Predicted'] <= 70:
            return 0.5
        else:
            return 0.0
    def low_gas_exchange(self):
        if self.df_dic_case['DLCO Predicted'] < 40:
            return 1.0
        elif self.df_dic_case['DLCO Predicted'] >= 40 and self.df_dic_case['DLCO Predicted'] <= 60:
            return 0.5
        else:
            return 0.0
    def prior_surgery(self):
        if self.df_dic_case['Prior Cardiothoracic Surgery'] == "Yes":
            return 1.0
        else:
            return 0.0
    def neoadjuvant_therapy(self):
        if self.df_dic_case['Preoperative Chemo - Current Malignancy'] == "Yes" or self.df_dic_case['Preoperative Thoracic Radiation Therapy'] == "Yes":
            return 1.0
        else:
            return 0.0
    def smoking_burden(self):
        if self.df_dic_case['Cigarette Smoking'] == "Current smoker":
            if self.df_dic_case['Pack-Years Of Cigarette Use'] < 20:
                return 0.68
            elif self.df_dic_case['Pack-Years Of Cigarette Use'] >= 20 and self.df_dic_case['Pack-Years Of Cigarette Use'] < 50:
                return 0.85
            else:
                return 1.0
        elif self.df_dic_case['Cigarette Smoking'] == "Past smoker (stopped more than 1 month prior to operation)":
            if self.df_dic_case['Pack-Years Of Cigarette Use'] < 20:
                return 0.17
            elif self.df_dic_case['Pack-Years Of Cigarette Use'] >= 20 and self.df_dic_case['Pack-Years Of Cigarette Use'] < 50:
                return 0.34
            else:
                return 0.51
        else:
            return 0.0
    def functional_status(self):
        if self.df_dic_case['ECOG Score'] == "0 - Fully active, able to carry on all pre-disease performance without restriction":
            return 0.0
        elif self.df_dic_case['ECOG Score'] == "1 - Restricted in physically strenuous activity but ambulatory and able to carry out work of a light or sedentary nature, e.g., light house work, offi" or self.df_dic_case['ECOG Score'] == "2 - Ambulatory and capable of all self-care but unable to carry out any work activities. Up and about more than 50'%' of waking hours":
            return 0.5
        else:
            return 1.0
    def systemic_risk(self):
        if self.df_dic_case['ASA Classification'] == "I":
            return 0.0
        elif self.df_dic_case['ASA Classification'] == "II":
            return 0.25
        elif self.df_dic_case['ASA Classification'] == "III":
            return 0.5
        elif self.df_dic_case['ASA Classification'] == "IV":
            return 0.75
        else:
            return 1.0
    def tumor_size(self):
        if self.df_dic_case['Tumor Size'] == "<3cm":
            return 0.25
        elif self.df_dic_case['Tumor Size'] == "3-5cm":
            return 0.5
        elif self.df_dic_case['Tumor Size'] == "5-7cm":
            return 0.75
        else:
            return 1.0
    def T_staging(self):
        if self.df_dic_case['Clinical Staging - Lung Cancer - T'] == "Tis":
            return 0.0
        elif self.df_dic_case['Clinical Staging - Lung Cancer - T'] == "T1" or self.df_dic_case['Clinical Staging - Lung Cancer - T'] == "T2":
            return 0.5
        else:
            return 1.0
    def N_staging(self):
        if self.df_dic_case['Clinical Staging - Lung Cancer - N'] == "N0":
            return 0.0
        elif self.df_dic_case['Clinical Staging - Lung Cancer - N'] == "N1":
            return 0.5
        else:
            return 1.0
    def M_staging(self):
        if self.df_dic_case['Clinical Staging - Lung Cancer - M'] == "M0":
            return 0.0
        else:
            return 1.0
    def procedure_burden(self):
        if self.df_dic_case['Procedure'] == "Lymphadenectomy":
            return 0.142
        elif self.df_dic_case['Procedure'] == "wedge":
            return 0.285
        elif self.df_dic_case['Procedure'] == "segmentectomy":
            return 0.426
        elif self.df_dic_case['Procedure'] == "unlisted":
            return 0.5
        elif self.df_dic_case['Procedure'] == "plication":
            return 0.568
        elif self.df_dic_case['Procedure'] == "decortication":
            return 0.71
        elif self.df_dic_case['Procedure'] == "lobectomy":
            return 0.852
        elif self.df_dic_case['Procedure'] == "chest wall excision":
            return 0.95
        else:
            return 1.0
    def age_pulmonary(self):
        if self.df_dic_case['Age'] >= 70 and (self.df_dic_case['FEV1 Predicted'] < 60 or self.df_dic_case['DLCO Predicted'] < 60):
            return 1.0
        elif (self.df_dic_case['Age'] >= 65 and self.df_dic_case['Age'] < 70) and (self.df_dic_case['FEV1 Predicted'] < 70 or self.df_dic_case['DLCO Predicted'] < 70):
            return 0.5
        else:
            return 0.0
    def age_surgical_burden(self):
        if self.df_dic_case['Age'] >= 70 and self.df_dic_case['Prior Cardiothoracic Surgery'] == "Yes":
            return 1.0
        elif (self.df_dic_case['Age'] >= 65 and self.df_dic_case['Age'] < 70) and self.df_dic_case['Prior Cardiothoracic Surgery'] == "Yes":
            return 0.5
        else:
            return 0.0
    def smoking_pulmonary(self):
        if self.df_dic_case['Cigarette Smoking'] == "Current smoker" and (self.df_dic_case['FEV1 Predicted'] < 60 or self.df_dic_case['DLCO Predicted'] < 60):
            return 1.0
        elif self.df_dic_case['Cigarette Smoking'] == "Past smoker (stopped more than 1 month prior to operation)" and (self.df_dic_case['FEV1 Predicted'] < 60 or self.df_dic_case['DLCO Predicted'] < 60):
            return 0.5
        else:
            return 0.0
    def procedure_pulmonary(self):
        if self.df_dic_case['Procedure'] in ["pneumonectomy", "chest wall excision"] and (self.df_dic_case['FEV1 Predicted'] < 60 or self.df_dic_case['DLCO Predicted'] < 60):
            return 1.0
        elif self.df_dic_case['Procedure'] in ["lobectomy", "decortication"] and (self.df_dic_case['FEV1 Predicted'] < 60 or self.df_dic_case['DLCO Predicted'] < 60):
            return 0.75
        elif self.df_dic_case['Procedure'] in ["unlisted", "segmentectomy", "plication"] and (self.df_dic_case['FEV1 Predicted'] < 60 or self.df_dic_case['DLCO Predicted'] < 60):
            return 0.5
        else:
            return 0.25
    def concept_building(self):
        self.extracting_raw_val()
        for case in tqdm(self.df_dic):
            self.df_dic_case       = self.df_dic[case]
            c1                     = self.age_burden()
            c2                     = self.low_bmi()
            c3                     = self.obesity()
            c4                     = self.low_bmi_high_age()
            c5                     = self.airflow_reserve()
            c6                     = self.low_gas_exchange()
            c7                     = self.prior_surgery()
            c8                     = self.neoadjuvant_therapy()
            c9                     = self.smoking_burden()
            c10                    = self.functional_status()
            c11                    = self.systemic_risk()
            c12                    = self.tumor_size()
            c13                    = self.T_staging()
            c14                    = self.N_staging()
            c15                    = self.M_staging()
            c16                    = self.procedure_burden()
            c17                    = self.age_pulmonary()
            c18                    = self.age_surgical_burden()
            c19                    = self.smoking_pulmonary()
            c20                    = self.procedure_pulmonary()
            self.concept_dic[case] = [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20]
        with open("Weak Concepts file for the train/val/test split", "w") as js:
            json.dump(self.concept_dic, js, indent = 4)

if __name__ == "__main__":
    concept_class = Concept_builder("train")
    concept_class.concept_building()
    concept_class = Concept_builder("val")
    concept_class.concept_building()
    concept_class = Concept_builder("test")
    concept_class.concept_building()