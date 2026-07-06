# SurgConcept-CT
**Codebase for anonymous submission to WACV 2027 R1**

In this paper, we present SurgConcept-CT, a clinically guided concept bottleneck multimodal framework for predicting postoperative complications probability after lung cancer surgery, using preoperative clinical data and CT imaging. The proposed framework consists of following parts:
- Clinical feature tokenization and encoder branch: consist of a tokenization layer that tokenizes clinical features' name, types, ordinal and numerical values and encodes them using a transformer encoder
- CT volume encoder and Regional pooling: Takes the preprocessed CT encoder as inputs and encodes them using frozen TANGERINE encoder, pooling the representation from it into an internal token grid of 64 coarse spatial regions, followed by projection through a trainable MLP projector 
- Clinically guided CT Regional Attention: Passing the clinical representations and CT representations through different linear layers provide us with **Q**, **K** and **V** matrices. This branch uses clinical representations to query CT regional tokens, providing us with attention weights over 64 CT regions. The final CT representation is obtained by weighted aggregation, representing a patient specific CT summary. with regional weighting conditioned on structured clinical context
- Concept predictor: Concatenates the clinical and CT representations obtained before and predict 20 weak concepts associated with patient
- Fusion and risk prediction: Takes in the concatenated clinical and CT representations, to predict the fused logit and adds it with concept logits to predict the final probability of postoperative risk
- Full architecture diagram is provided below, with details in the paper

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
- We evaluate SurgConcept-CT on **LungComp-CT**, a private real world cohort of lung cancer surgery patients treated at a prominent cancer research hospital between 2009 and 2023
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

## Training
- SurgConcept-CT can be trained using train.py script on the training split of LungComp-CT and validates the performance per epoch on the validation split
- SurgConcept-CT was trained in 2 stages: clinical only and full model
- Script runs for both stages of training
- Hyperparameters regarding the training settings are provided in the paper

## Testing and Evaluation
- Trained model checkpoint (provided above) can be evaluated on the testing split using inference.py script

## Results
### Quantitative results

#### Performance across baselines and SurgConcept-CT
|Model | Inputs | AUC(%) | TAR(%)@FAR=0.2 | TAR(%)@FAR=0.3 |
| :---: | :---: | :---: | :---: | :---: |
|Logistic Regression | clinical only | 75.11   | 57.32 | 71.95   |
|Gradient Boosting Classifier | clinical only  | 70.77  | 45.12 | 68.29  |
|XGBoost | clinical only  | 66.69  | 41.46  | 54.88  |
|Random Forest Classifier | clinical only   | 72.29 | 56.10   | 64.63  |
|SVC | clinical only   | 71.23 | 46.34   | 54.88  |
|TabTransformer | clinical only   | 74.35 | 52.44   | 73.17  |
|FTTransformer | clinical only   | 73.79 | 59.76   | 70.73  |
|Merlin | CT only   | 55.96 | 24.39   | 41.46  |
|TANGERINE | CT only   | 66.24 | 35.37   | 50.00  |
|Merlin | CT + clinical summaries   | 70.52 | 47.56   | 60.98  |
|M3D-LaMed | CT + clinical summaries   | 70.81 | 45.12   | 62.19  |
|RadFM | CT + clinical summaries   | 73.69 | 53.66   | 68.29  |
|SurgConcept-CT | clinical + CT   | **76.85** | **64.63**   | **74.39**  |

#### ROC for performance comparison
![ROC](https://github.com/anonymouspeerblind/SurgConcept-CT/blob/main/combined_ROC.png)

### Qualitative CT attention results
![ct-attention](https://github.com/anonymouspeerblind/SurgConcept-CT/blob/main/attention_overlay_montage.png)
Clinically guided CT attention overlay for a test split patient, who developed a postoperative pulmonary complication. Axial slices from the reconstructed preoperative CT volume are shown with the model’s normalized CT attention scores overlaid. Each colored block corresponds to one regional CT token, with warmer colors indicating regions assigned with greater attention by the clinically guided CT attention module during risk prediction. This illustrates model level regional importance and not voxel level disease localization.


## Ablation study
|Clinical | CT Volume | AUC(%) | TAR(%)@FAR=0.2 | TAR(%)@FAR=0.3 |
| :---: | :---: | :---: | :---: | :---: |
| **&check;** | **&hyphen;** | 74.61 | 52.44 | 70.73 |
| **&check;** | **&check;** | 76.85 | 64.63 | 74.39 |

## Contact
For more information, feel free to reach us at anonymouspeerblind@gmail.com

## License
SurgConcept-CT is CC-BY-NC 4.0 licensed, as found in the LICENSE file.
