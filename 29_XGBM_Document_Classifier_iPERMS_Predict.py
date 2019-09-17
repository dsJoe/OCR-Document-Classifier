## Modified on 20180511
import pytesseract
import subprocess
import os
import sys
import re
import traceback
import shutil
from PIL import Image, ImageEnhance, ImageFilter, ImageSequence
import glob
import pandas as pd
import pickle
from datetime import datetime,date


## Get an Image files from a directory
def getImageFiles(source_dir):

    result_list = []
    for files in os.listdir(source_dir):
        if re.search("(.jpg|.png|.tif|.jpeg)$", files):
        
            result_list.append(os.path.join(source_dir,files))
        
    return result_list


## Rotate an image 
def getRotation(filenames):
    tmp = subprocess.Popen(['tesseract', filenames, '-', '-psm', '0'],stdout= subprocess.PIPE, stderr = subprocess.PIPE).communicate()[1]
    
    for i in tmp.split("\n"):
        if i.split(":")[0] == 'Orientation in degrees':
            return int(i.split(":")[1].strip())
    return 0
   
## split an images with more than one page        
def splitFiles(filenames_list, source_dir):
    first_regex = re.compile("\w*(.jpg|.png|.tif|.jpeg)$")
    second_regex = re.compile("\w*(?=(.jpg|.png|.tif|.jpeg)$)")
    third_regex = re.compile("(.jpg|.png|.tif|.jpeg)$")
    for files in filenames_list:
        img = Image.open(files)
        for i, page in enumerate(ImageSequence.Iterator(img)):
            page.save(first_regex.sub('',files) + "Image_split_output/" + 
                      second_regex.search(files).group(0)+
                      "_page%d%s"%(i, third_regex.search(files).group(0)))
            
            
## create a directory to save the images            
def createImageSplitDirectory(source_dir):
    if os.path.isdir(source_dir+"/Image_split_output") is False:
        os.mkdir(source_dir+"/Image_split_output")

        
    
## extract a text from images   
def startProcessing(source_dir, target_dir):    
        
    for files in os.listdir(source_dir+"/Image_split_output"):
          
        im = Image.open(source_dir+"/Image_split_output/"+files)
    
        im = im.filter(ImageFilter.MedianFilter())
        #enhancer = ImageEnhance.Contrast(im)
        #im = enhancer.enhance(2)
        #im = im.convert('1')
        result= pytesseract.image_to_string(im.rotate(360-getRotation(source_dir+"/Image_split_output/"+files)))
        z= removePunctuation(result)
        
        ##uncomment Print to view the results
        #print z
        with open(target_dir + "/" + files+'_.txt', 'w+') as result_file:
            result_file.write(z)

## cleaning the extracted content     
def removePunctuation(opencv1_text):
    rmnewline = opencv1_text.replace('\n', ' ').replace('\r',' ').strip()
    return re.sub('[ \t]+',' ',rmnewline).encode('utf8')

def cleandocs(text):
        nospc = re.sub(r'[^a-z0-9\s]',' ',text.lower())
        noalph = re.sub(r'\b\w{1,1}\b', '',nospc)
        return noalph.encode('utf8')


def classify_doc(src_dir):
    
    global corpus,process_type

    #content extraction for multiple files
    for file_name in os.listdir(src_dir):
        filename = (os.path.join(src_dir,file_name))
        with open(filename) as f_input:
            content = cleandocs(f_input.read().replace('\n', ' '))
            df = pd.DataFrame({
                               'filename':[filename],
                               'content': [content]
                               })
            corpus = corpus.append(df)

    #word level tf-idf    
    tf_model = pickle.load(open(model_path+'/'+'tfid_vect.sav', 'rb'))
    xinput_tfidf =  tf_model.transform(corpus['content'])
    
    #loading model
    model = pickle.load(open(model_path+'/'+model_file_name, 'rb'))
    
    #prediction
    encoder_model = pickle.load(open(model_path+'/'+'encoder.sav', 'rb'))
    predictions = model.predict(xinput_tfidf.tocsc())
    columns_prob = encoder_model.inverse_transform([0,1,2,3,4,5])
        
    predictions = encoder_model.inverse_transform(predictions)
    preds = pd.DataFrame(predictions)
    preds.columns = ['prediction']
    filename_df = corpus['filename']
      
    filename_df.reset_index(drop=True,inplace=True)
    
    df_final = pd.concat([filename_df,preds], axis=1)
    
    ## probabilities
    probs = model.predict_proba(xinput_tfidf.tocsc())
    
    probs = pd.DataFrame(probs)
    probs.columns = columns_prob
    
    probs['max_value'] = probs.max(axis=1)

    #print(preds)
    df_final = pd.concat([filename_df,preds,probs], axis=1)
    
    df_final = df_final.fillna(0)
    
    df_final['prediction'] = df_final.apply(lambda x : 'NOT CLASSIFIED'                                   if x['max_value'] < confidence_level else x['prediction'],axis=1)
    
    #print(df_final)
   
    target_filename = ('/iperms_classification_output_%s.csv') % (date.today())
    
    df_final = df_final[['filename','prediction','max_value']]
    df_final = df_final.rename(columns={'max_value':'probability'})
    
    print(df_final)
    
    df_final.to_csv(output_dir+target_filename, index=None, sep=',', mode='w',encoding='utf-8')


            
def main():
   
    try:
        global filename,model_path,model_file_name,corpus,output_dir,process_type,confidence_level

        # extracting commandline arguments 
        print('Number of arguments:', len(sys.argv))

        if sys.argv[1] == '-f':
            print('cosidering the parameters from the main function')
            model_path = '/home/jdas/prj/endzone/python/notebooks/joe/iperms/train/model'
            model_file_name = 'iperms_calssification_model_xgb_2018-05-11.sav'
            #src_dir = '/home/jdas/prj/endzone/python/notebooks/joe/iperms/test'
            output_dir = '/home/jdas/prj/endzone/python/notebooks/joe/iperms/output'
            confidence_level = .70
            source_dir = "/home/jdas/prj/endzone/python/notebooks/joe/iperms/train/images/"
            target_dir = "/home/jdas/prj/endzone/python/notebooks/joe/iperms/train/extracts/"

        elif len(sys.argv) != 4 :
            sys.exit('invalid number of input arguments')

        else:
            model_path = '/root/iperms/model'
            model_file_name = 'iperms_calssification_model_xgb_2018-05-11.sav'
            source_dir = sys.argv[1]
            target_dir = sys.argv[2]
            output_dir = sys.argv[3]
            confidence_level = .70
            print('All arguments are correct....proceeding to extraction and classification')
    
    except (SystemExit,KeyboardInterrupt):
            print('stopping the script gracefully....',sys.exc_info()[0:2]) 
    
    except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            msg = ''.join('!! ' + line for line in lines)
            print (msg)
    
    try:
        files_list = getImageFiles(source_dir)
        print files_list
            
    except IOError as exp:
        print "----Exeption Occured While Reading Files From Target Directory----"
        print "Exception NO:",exp.errno
        print "Exceprtion Message:",exp.message
        
    
    if len(files_list) == 0:
        print "----There are no files in Target Directory---"
        sys.exit(0)
      
    createImageSplitDirectory(source_dir)
    
    try:
        splitFiles(files_list, source_dir)
    
    except IOError as exp:
        print "----Exeption Occured While Writing Files Into Image_Split_Dir----"
        print "Exception NO:",exp.errno
        print "Exceprtion Message:",exp.message
        
    startProcessing(source_dir, target_dir)
    shutil.rmtree(source_dir+"/Image_split_output")
    
    # Calling classification model
    try:      
        corpus = pd.DataFrame()
        classify_doc(target_dir)
    
    except (SystemExit,KeyboardInterrupt):
        print('stopping the script gracefully....',sys.exc_info()[0:2]) 
    
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        msg = ''.join('!! ' + line for line in lines)
        print (msg)   

if __name__ == "__main__":
    try:
        ## main( "path for the image folder", "path a target folder" )
        #main("/home/skaushik/iPERMS/DEMO/Forms_Image/","/home/skaushik/iPERMS/DEMO/Text_Files/" )
        main()
        
    except:
        print '------------Exception in main block---------------' 
        exc_type, exc_value, exc_traceback = sys.exc_info()
        lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        print  '\n'.join('!! ' + line for line in lines)
