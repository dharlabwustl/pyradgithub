
import os, sys, errno, shutil, uuid,subprocess,csv,json
import math,inspect
import glob
import re,time
import requests
import pandas as pd
import nibabel as nib
import numpy as np
# import pydicom as dicom
import pathlib
import argparse,xmltodict
from xnatSession import XnatSession
# from redcapapi_functions import *
catalogXmlRegex = re.compile(r'.*\.xml$')
XNAT_HOST_URL=os.environ['XNAT_HOST']  #'http://snipr02.nrg.wustl.edu:8080' #'https://snipr02.nrg.wustl.edu' #'https://snipr.wustl.edu'
XNAT_HOST = XNAT_HOST_URL # os.environ['XNAT_HOST'] #
XNAT_USER = os.environ['XNAT_USER']#
XNAT_PASS =os.environ['XNAT_PASS'] #
api_token=os.environ['REDCAP_API']
xnatSession = XnatSession(username=XNAT_USER, password=XNAT_PASS, host=XNAT_HOST)
class arguments:
    def __init__(self,stuff=[]):
        self.stuff=stuff
def download_a_singlefile_with_URIString(url,filename,dir_to_save):
    print("url::{}::filename::{}::dir_to_save::{}".format(url,filename,dir_to_save))
    # xnatSession = XnatSession(username=XNAT_USER, password=XNAT_PASS, host=XNAT_HOST)
    xnatSession.renew_httpsession()
    # command="echo  " + url['URI'] + " >> " +  os.path.join(dir_to_save,"test.csv")
    # subprocess.call(command,shell=True)
    response = xnatSession.httpsess.get(xnatSession.host +url) #/data/projects/ICH/resources/179772/files/ICH_CTSESSIONS_202305170753.csv") #
    #                                                       # "/data/experiments/SNIPR02_E03548/scans/1-CT1/resources/147851/files/ICH_0001_01022017_0414_1-CT1_threshold-1024.0_22121.0TOTAL_VersionDate-11302022_04_22_2023.csv") ## url['URI'])
    zipfilename=os.path.join(dir_to_save,filename ) #"/data/projects/ICH/resources/179772/files/ICH_CTSESSIONS_202305170753.csv")) #sessionId+scanId+'.zip'
    with open(zipfilename, "wb") as f:
        for chunk in response.iter_content(chunk_size=512):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
    # xnatSession.close_httpsession()
    return zipfilename

def get_selected_scan_info(SESSION_ID, dir_to_save):
    # Log failure message for debugging
    command = f"echo failed at each_row_id: {inspect.stack()[0][3]} >> output/error.txt"
    subprocess.call(command, shell=True)

    # Define the URI and URL for the request
    URI = f'/data/experiments/{SESSION_ID}'
    url = f"{URI}/resources/NIFTI_LOCATION/files?format=json"

    # Initialize XNAT session
    xnatSession = XnatSession(username=XNAT_USER, password=XNAT_PASS, host=XNAT_HOST)
    xnatSession.renew_httpsession()

    try:
        # Make a GET request to the XNAT server
        response = xnatSession.httpsess.get(xnatSession.host + url)
        response.raise_for_status()  # Raise an error for unsuccessful responses
        metadata_masks = response.json()['ResultSet']['Result']

        # Convert metadata to a DataFrame
        df_scan = pd.read_json(json.dumps(metadata_masks))

        # Log additional debug information
        command = f"echo failed_1 at each_row_id: {df_scan['URI'].iloc[0]} >> output/error.txt"
        subprocess.call(command, shell=True)

        # Download the file specified by the URI
        download_a_singlefile_with_URIString(
            str(df_scan['URI'].iloc[0]),
            os.path.basename(str(df_scan['URI'].iloc[0])),
            dir_to_save
        )

        # Read the downloaded file to extract SCAN_ID and SCAN_NAME
        nifti_location = pd.read_csv(
            os.path.join(dir_to_save, os.path.basename(str(df_scan['URI'].iloc[0])))
        )
        SCAN_ID = str(nifti_location.loc[nifti_location.index[0], 'ID'])
        SCAN_NAME = str(nifti_location.loc[nifti_location.index[0], 'Name'])

        # Log success message
        command = f"echo passed at each_row_id: {inspect.stack()[0][3]} >> output/error.txt"
        subprocess.call(command, shell=True)

        return SCAN_ID, SCAN_NAME

    except Exception as e:
        # Log exception details for debugging
        command = f"echo exception at {inspect.stack()[0][3]}: {str(e)} >> output/error.txt"
        subprocess.call(command, shell=True)
        raise e

    finally:
        # Close the XNAT session
        xnatSession.close_httpsession()

def download_an_xmlfile_with_URIString_func(session_ID,filename,dir_to_save): #url,filename,dir_to_save):
    returnvalue=0

    try:
        # session_ID=str(args.stuff[1])
        # filename=str(args.stuff[2])
        # dir_to_save=str(args.stuff[3])
        subprocess.call('echo working:' +session_ID+' > /workingoutput/testatul.txt',shell=True)
        subprocess.call('echo working:' +filename+' > /workingoutput/testatul.txt',shell=True)
        subprocess.call('echo working:' +dir_to_save+' > /workingoutput/testatul.txt',shell=True)
        print("url::{}::filename::{}::dir_to_save::{}".format(session_ID,filename,dir_to_save))
        xnatSession = XnatSession(username=XNAT_USER, password=XNAT_PASS, host=XNAT_HOST)
        xnatSession.renew_httpsession()

        # command="echo  " + url['URI'] + " >> " +  os.path.join(dir_to_save,"test.csv")
        # subprocess.call(command,shell=True)
        # https://snipr.wustl.edu/app/action/XDATActionRouter/xdataction/xml_file/search_element/xnat%3ActSessionData/search_field/xnat%3ActSessionData.ID/search_value/SNIPR02_E03847
        url='/app/action/XDATActionRouter/xdataction/xml_file/search_element/xnat%3ActSessionData/search_field/xnat%3ActSessionData.ID/search_value/'+str(session_ID)  ##+'/popup/false/project/ICH'
        subprocess.call("echo " + "I url AT ::{}  >> /workingoutput/error.txt".format(xnatSession.host +url) ,shell=True )
        xmlfilename=os.path.join(dir_to_save,filename )
        try:
            response = xnatSession.httpsess.get(xnatSession.host +url) #/data/projects/ICH/resources/179772/files/ICH_CTSESSIONS_202305170753.csv") #
            num_files_present=0
            subprocess.call("echo " + "I response AT ::{}  >> /workingoutput/error.txt".format(response) ,shell=True )
            metadata_masks=response.text #json()['ResultSet']['Result']
            f = open(xmlfilename, "w")
            f.write(metadata_masks)
            f.close()
            # if response.status_code != 200:
        except:
            command='curl -u '+ XNAT_USER +':'+XNAT_PASS+' -X GET '+ xnatSession.host +url + ' > '+ xmlfilename
            subprocess.call(command,shell=True)
        # xnatSession.close_httpsession()
        # return num_files_present



        # zipfilename=os.path.join(dir_to_save,filename ) #"/data/projects/ICH/resources/179772/files/ICH_CTSESSIONS_202305170753.csv")) #sessionId+scanId+'.zip'
        # with open(zipfilename, "wb") as f:
        #     for chunk in response.iter_content(chunk_size=512):
        #         if chunk:  # filter out keep-alive new chunks
        #             f.write(chunk)
        xnatSession.close_httpsession()
        returnvalue=1
        subprocess.call("echo " + "I PASSED AT ::{}  >> /workingoutput/error.txt".format(inspect.stack()[0][3]) ,shell=True )
        print("I PASSED AT ::{}".format(inspect.stack()[0][3]))
    except:

        subprocess.call("echo " + "I FAILED AT ::{}  >> /workingoutput/error.txt".format(inspect.stack()[0][3]) ,shell=True )
        print("I PASSED AT ::{}".format(inspect.stack()[0][3]))
    return returnvalue



def get_metadata_session(sessionId,outputfile="NONE.csv"):
    url = ("/data/experiments/%s/scans/?format=json" %    (sessionId))
    xnatSession = XnatSession(username=XNAT_USER, password=XNAT_PASS, host=XNAT_HOST)
    xnatSession.renew_httpsession()
    response = xnatSession.httpsess.get(xnatSession.host + url)
    xnatSession.close_httpsession()
    metadata_session=response.json()['ResultSet']['Result']
    metadata_session_1=json.dumps(metadata_session)
    df_scan = pd.read_json(metadata_session_1)
    df_scan.to_csv(outputfile,index=False)
    return metadata_session