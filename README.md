# SurgConcept-CT
**Codebase for anonymous submission to WACV 2027 R1**

In this paper, we present SurgConcept-CT, a clinically guided concept bottleneck multimodal framework for predicting postoperative complications probability after lung cancer surgery, using preoperative clinical data and CT imaging. The proposed framework consists of following parts:
- Clinical data encoder branch: 

![SurgConcept-CT architecture](https://github.com/anonymouspeerblind/SurgConcept-CT/blob/main/surgconcept_ct_architecture.png)

## Installation and environment
- In the build.sh file, insert path of your working code directory, path of directory containing your data and a big scratch folder path in PROJECT_DIR_1, PROJECT_DIR_2 and PROJECT_DIR_3 respectively
- Change the path of hooks-dir, huggingface home, hub cache and XET cache paths in build.sh to your appropriate paths
- Change the workspace path in build.sh
- DOCKERFILE has all the required packages, change the paths in the DOCKERFILE as well
- Build your docker environment as:
```
chmod +x build.sh
./build.sh
```
Note: We are using podman for building this environment.

## Preparing and Preprocessing data
- We evaluate SurgConcept-CT on LungComp-CT, a private real world cohort of lung cancer surgery patients treated at a prominent cancer research hospital between 2009 and 2023
- Due to patient privacy and institutional restrictions, LungComp-CT cannot be publicly released
- To the best of our knowledge, no publicly available dataset currently matches the structure of our dataset or addresses the same downstream task of predicting the probability of postoperative pulmonary complications after lung cancer surgery
- The dataset contains 3,277 patients and is split into 2,719 training, 279 validation and 279 testing patients respectively
- In case you acquire a dataset, which has the same structure as LungComp-CT, use preprocessing files in "preprocessing" folder to preprocess your data
- For preprocessing clinical data, use clinical_data_find_process.py to standardize and save the input worthy clinical data
- Use cropping_preprocessing_cts.py file to preprocess the original CT volumes by cropping it around thoracic region using TotalSegmentator, resampling, windowing, normalization and saving them as a fixed shape of (256, 256, 256), which can then be processed through frozen TANGERINE encoder
- Use weak_concept_builder.py file, to create weak concepts for all datapoints in all three splits

## Pretrained Checkpoints
- For clinical only model: [Link](https://drive.google.com/file/d/1z3m7zFv8-sMateoP0oJlMinrLcQ0iIQq/view?usp=sharing)
- For full SurgConcept-CT model: [Link](https://drive.google.com/file/d/1sDQGqHiYjTYqu7kqaRIkMJwElacbHTy7/view?usp=sharing)
