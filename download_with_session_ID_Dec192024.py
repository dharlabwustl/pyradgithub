
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
def get_info_from_xml(xmlfile):
    try:
        # session_id=args.stuff[1]
        # xmlfile=args.stuff[2]
        subprocess.call("echo " + "I xmlfile AT ::{}  >> /workingoutput/error.txt".format(xmlfile) ,shell=True )
        with open(xmlfile, encoding="utf-8") as fd:
            xmlfile_dict = xmltodict.parse(fd.read())

        project_name=''
        subject_name=''
        session_label=''
        acquisition_site_xml=''
        acquisition_datetime_xml=''
        scanner_from_xml=''
        body_part_xml=''
        kvp_xml=''
        try:
            session_label=xmlfile_dict['xnat:CTSession']['@label']
        except:
            pass
        try:
            project_name=xmlfile_dict['xnat:CTSession']['@project']
        except:
            pass
        try:
            subject_name=str(xmlfile_dict['xnat:CTSession']['xnat:dcmPatientName'])
        except:
            pass
        # Acquisition site
        try:
            acquisition_site_xml=xmlfile_dict['xnat:CTSession']['xnat:acquisition_site']
        except:
            pass
        try:
            date_split=str(xmlfile_dict['xnat:CTSession']['xnat:date']).split('-')
            columnvalue_1="/".join([date_split[1],date_split[2],date_split[0]])
            columnvalue_2=":".join(str(xmlfile_dict['xnat:CTSession']['xnat:time']).split(':')[0:2])
            acquisition_datetime_xml=columnvalue_1+" "+ columnvalue_2
        except:
            pass
        columnvalue_1=""
        columnvalue_2=""
        try:
            for xx in range(len(xmlfile_dict['xnat:CTSession']['xnat:scans']['xnat:scan'])):
                if xx > 1:
                    try:
                        columnvalue_1= xmlfile_dict['xnat:CTSession']['xnat:scans']['xnat:scan'][xx]['xnat:scanner']['@manufacturer']
                        if len(columnvalue_1)>1:
                            break
                    except:
                        pass
            for xx in range(len(xmlfile_dict['xnat:CTSession']['xnat:scans']['xnat:scan'])):
                if xx > 1:
                    try:
                        columnvalue_2=  xmlfile_dict['xnat:CTSession']['xnat:scans']['xnat:scan'][xx]['xnat:scanner']['@model']
                        if len(columnvalue_2)>1:
                            break
                    except:
                        pass

            scanner_from_xml= columnvalue_1 + " " + columnvalue_2
        except:
            pass
        ################
        try:
            for xx in range(len(xmlfile_dict['xnat:CTSession']['xnat:scans']['xnat:scan'])):
                if xx>1:
                    try:
                        body_part_xml=xmlfile_dict['xnat:CTSession']['xnat:scans']['xnat:scan'][xx]['xnat:bodyPartExamined']
                        # fill_datapoint_each_sessionn_1(identifier,columnname,columnvalue,csvfilename)
                        if len(body_part_xml)>3:
                            break
                    except:
                        pass

        except:
            pass
        ###############
        ###############

        try:
            for xx in range(len(xmlfile_dict['xnat:CTSession']['xnat:scans']['xnat:scan'])):
                if xx>1:
                    try:
                        kvp_xml=xmlfile_dict['xnat:CTSession']['xnat:scans']['xnat:scan'][xx]['xnat:parameters']['xnat:kvp']
                        # fill_datapoint_each_sessionn_1(identifier,columnname,columnvalue,csvfilename)
                        if len(kvp_xml)>1:
                            break
                    except:
                        pass
            # columnvalue=xmlfile_dict['xnat:CTSession']['xnat:scans']['xnat:scan'][0]['xnat:parameters']['xnat:kvp']
            # fill_datapoint_each_sessionn_1(identifier,columnname,columnvalue,csvfilename)
        except:
            subprocess.call("echo " + "I PASSED AT ::{}  >> /workingoutput/error.txt".format(inspect.stack()[0][3]) ,shell=True )
            pass
        ###############

        subprocess.call("echo " + "I PASSED AT ::{}  >> /workingoutput/error.txt".format(inspect.stack()[0][3]) ,shell=True )
        subprocess.call("echo " + "I PASSED AT project_name::{}  >> /workingoutput/error.txt".format(project_name) ,shell=True )
        subprocess.call("echo " + "I PASSED AT subject_name::{}  >> /workingoutput/error.txt".format(subject_name) ,shell=True )
        subprocess.call("echo " + "I PASSED AT session_label::{}  >> /workingoutput/error.txt".format(session_label) ,shell=True )
        subprocess.call("echo " + "I PASSED AT acquisition_site_xml::{}  >> /workingoutput/error.txt".format(acquisition_site_xml) ,shell=True )
        subprocess.call("echo " + "I PASSED AT ::{}  >> /workingoutput/error.txt".format(inspect.stack()[0][3]) ,shell=True )
        subprocess.call("echo " + "I PASSED AT acquisition_datetime_xml::{}  >> /workingoutput/error.txt".format(acquisition_datetime_xml) ,shell=True )
        subprocess.call("echo " + "I PASSED AT scanner_from_xml::{}  >> /workingoutput/error.txt".format(scanner_from_xml) ,shell=True )
        subprocess.call("echo " + "I PASSED AT body_part_xml::{}  >> /workingoutput/error.txt".format(body_part_xml) ,shell=True )
        subprocess.call("echo " + "I PASSED AT kvp_xml::{}  >> /workingoutput/error.txt".format(kvp_xml) ,shell=True )
    except:
        # print("I FAILED AT ::{}".format(inspect.stack()[0][3]))
        subprocess.call("echo " + "I FAILED AT ::{}  >> /workingoutput/error.txt".format(inspect.stack()[0][3]) ,shell=True )
        pass
    return   project_name,subject_name, session_label,acquisition_site_xml,acquisition_datetime_xml,scanner_from_xml,body_part_xml,kvp_xml
def downloadfile_withasuffix(sessionId,scanId,output_dirname,resource_dirname,file_suffix):
    try:
        print('sessionId::scanId::resource_dirname::output_dirname::{}::{}::{}::{}'.format(sessionId,scanId,resource_dirname,output_dirname))
        url = (("/data/experiments/%s/scans/%s/resources/"+resource_dirname+"/files/") % (sessionId, scanId))
        df_listfile=listoffile_witha_URI_as_df(url)
        for item_id, row in df_listfile.iterrows():
            if file_suffix in str(row['URI']) : ##.str.contains(file_suffix):
                download_a_singlefile_with_URIString(row['URI'],row['Name'],output_dirname)
                print("DOWNLOADED ::{}".format(row))
        return True
    except Exception as exception:
        print("FAILED AT ::{}".format(exception))
        pass
    return  False

def listoffile_witha_URI_as_df(URI):
    # xnatSession = XnatSession(username=XNAT_USER, password=XNAT_PASS, host=XNAT_HOST)
    xnatSession.renew_httpsession()
    # print("I AM IN :: listoffile_witha_URI_as_df::URI::{}".format(URI))
    response = xnatSession.httpsess.get(xnatSession.host + URI)
    # print("I AM IN :: listoffile_witha_URI_as_df::URI::{}".format(URI))
    num_files_present=0
    df_scan=[]
    if response.status_code != 200:
        # xnatSession.close_httpsession()
        return num_files_present
    metadata_masks=response.json()['ResultSet']['Result']
    df_listfile = pd.read_json(json.dumps(metadata_masks))
    # xnatSession.close_httpsession()
    return df_listfile


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