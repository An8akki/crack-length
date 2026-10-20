
--- PAGE 1 ---
READ ME FILE FOR SDNET2021: Annotated NDE dataset for Structural Defects. 
The folders contain three types of non-destructive evaluation (NDE) data which are annotated using ground truth 
information collected from five in-service reinforced concrete bridge decks. The NDE data collected in this study 
are Impact Echo (IE), Ground Penetrating Radar (GPR), and Infrared Thermography (IRT). The dataset was 
annotated using three classes in accordance with bridge deck repair: Class 1  No Delamination; Class 2 
Delamination (delamination above top bar mat), and Class 3 Delamination (delamination below top bar mat).  
Name of parent folder (Dataset) – Structural Defect Dataset 2021 (SDNET2021) 
Name of Sub-folders- There are five (5) sub-folders with its contents described below: 
• 1. Ground Truth 
This folder contains eleven (11) files:  five (5) DWG cad files, and one (1) combined pdf file format. These files 
show the location of Class 2 and Class 3 delamination on the investigated bridge decks.  
• 2. IE and GPR TEST POINTS 
This folder contains ten (10) dwg CAD files for GPR and IE signals showing the test points, regions, and 
locations. The IE and GPR are plotted separately on the same layouts for each bridge decks. 206 GPR files 
containing 663,102 signals and 1,936 IE test point signals were collected for the study and shown in their 
respective folders. For the GPR data, the raw signals are directly annotated while the IE signals are annotated on 
a separate Ms excel workbook automatically using MATLAB 2020b program software.  
• 3. Impact Echo Data 
This folder contains two (2) sub-folders: Impact Echo Field Data and Impact Echo Annotated Data. 
a) 01-Impact Echo Field Data: This contains five (5) sub-folders. 
The five (5) sub-folders contain the raw data of the IE points collected on site for each of the bridges. 
Each of the 5 sub-folders contains sub-folders which contains the regions of test points A, B, C, D as 
appropriate. These regions (10’x10’) are plotted on the deck layout and shown in the report and in the 
‘IE and GPR TEST POINTS’ folder mentioned above. The raw data was provided as raw data for 
referencing its annotation which has been provided in the next sub-folder. 
b) 02-Impact Echo Annotated Data: This folder contains a Ms. Excel workbook format with name “ND 
Bridge_IE_ANNOTATION”. The workbook contains Five (5) sheets, each sheet for each of the 
bridges. The columns of these sheets are Bridge, Date, File Number, FileName_LVM, 
Origin_X_Offset_ft, Origin_Y_Offset_ft, Origin, Local_Y_ft, Data Quality (Inspector) and sub-
surface delamination (Delamination Class). This file must be read along with the raw files discussed in 
the field data above to identify the correct annotation of each of the IE signals.  An excerpt of the IE 
annotated data is shown in the Figure 1. 
--- PAGE 2 ---
 
 
 
Figure 1. IE Annotation workbook and annotation format 
• 4. GPR data 
This folder contains five (5) sub-folders of annotated GPR Data for each investigated bridge deck. Each folder 
contains signal data collected for each bridge deck while each file contains several hundreds and thousands of 
rows and columns. For instance, Figure 2 is an excerpt of file 001 (GPR signal 01) for Park River median bridge 
containing 16,383 amplitudes and 512 rows of time increment. The (x,y) coordinates of the signal amplitudes 
are shown and can be referenced with the test points in ‘IE and GPR TEST POINTS’ folder 2. In addition, each 
amplitude signal is annotated according to the class of delamination.  

--- PAGE 3 ---
 
Figure 2.GPR Annotation for signal 01 
• 5. Infrared Thermography data. 
This contains five (5) sub-folders including IRT stitched maps for each bridge. Each folder contains two (2) 
files each for: ‘Original images’ and ‘Annotated images. The ‘original images’ is a stitched map prior to 
annotation while the other folders named ‘Annotated images’ contains pixel annotated based on the class of 
delamination, 1, 2 or 3 depicted by distinct color codes; Class 1-No Delamination – (deck’s greyish color), 
Class 2 Delamination (the green component of RGB color space is 255) and Class 3 Delamination (the red 
component of the RGB color space is 255). In total, there are 5 each of original and annotated IRT images 
in the folder. 
