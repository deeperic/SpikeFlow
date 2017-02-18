import os
from shutil import copyfile
import cv2

aggregatedPath = 'train'
trainPath = 'train'
valPath = 'val'
imagesPath = ['cheung', 'chin2', 'chiu', 'cup', 'ho', 'ka', 'ma', 'sheung', 'shi', 'yi', 'sin', 'ngou', 'lai']


labelsFileTrain = './trainLabels.csv'
labelsFileVal = './valLabels.csv'


numVal = 160 #200 images, 80%  training, 20% validation

f = open(labelsFileTrain,'w')
f2 = open(labelsFileVal,'w')

aggregatedPath = trainPath #init with train path
writeFile = f

count = 0
eachClassCount = 0

for imagedir in imagesPath:  
    print(imagedir)    
    aggregatedPath = trainPath #init with train path
    writeFile = f
    eachClassCount = 0
        
    for fileName in os.listdir(imagedir):
        print(fileName)
        if(fileName == ".DS_Store"):
            continue
        filename_split = os.path.splitext(fileName)
        
        if eachClassCount >= numVal: #validation files
        	aggregatedPath = valPath
        	writeFile = f2
        	
        oldFileName = imagedir + "/" + fileName
        newFileOnlyName = '{0:05d}'.format(count) + "." + "png"
        newFileName = "./" + aggregatedPath + "/" + newFileOnlyName
        print(oldFileName)
        print(newFileName)
        
        copyfile(oldFileName, newFileName)
        writeFile.write('{0:05d}'.format(count) + "," + imagedir + '\n')
        
        
        count = count + 1
        eachClassCount = eachClassCount + 1
        
        
f.close()
f2.close()

def resizeme(path):

  #resize 32x32
  for fileName in os.listdir(path):
    if(fileName == ".DS_Store"):
      continue
    print("resize:" + fileName)
    image = cv2.imread(path + "/" + fileName)
    resized_image = cv2.resize(image, (32, 32)) 
    cv2.imwrite(path + "/" + fileName, resized_image)



resizeme(trainPath)
resizeme(valPath)
