# Vinam



## Introduction **TODO**

## Install

### Windows
1. Check python Version. Python Version used during development: **3.10.9**
2. Check pip Version. Pip Version: **22.3.1**
3. Clone Repository
4. Open CMD in the directory you cloned it
5. Run following command to get all dependencies:

```
pip install -r requirements.txt
```
*6. Optional* If you receive an error during the installation of **tensorflow** try this in Windows Powershell as an Admin:

```
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
-Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```
## Run

After you have successully installed all dependencies you can execute the programm:

```
cd vinam
python main.py
```