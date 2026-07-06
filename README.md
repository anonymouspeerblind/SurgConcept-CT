# SurgConcept-CT
**Anonymous submission to WACV 2027 R1**

![SurgConcept-CT architecture](https://github.com/anonymouspeerblind/SurgConcept-CT/blob/main/surgconcept_ct_architecture.png)

## Installations
- In the build.sh file, insert path of your working code directory, path of directory containing your data and a big scratch folder path in PROJECT_DIR_1, PROJECT_DIR_2 and PROJECT_DIR_3 respectively
- Change the path of hooks-dir, huggingface home, hub cache and XET cache paths in build.sh to your appropriate paths
- Change the workspace path in build.sh
- DOCKERFILE has all the required packages that are required, change the paths in the DOCKERFILE as well
- Build your docker environment as:
```
chmod +x build.sh
./build.sh
```
