from download_with_session_ID_Dec192024 import *
import sys,os,glob,subprocess
# print("I AM HERE")
# Example usage if run directly
def call_pyradiomics(SESSION_ID,file_output_dir):
    SCAN_ID,SCAN_NAME=get_selected_scan_info(SESSION_ID,file_output_dir)
    print(SCAN_ID)
    return 1
if __name__ == "__main__":
    # import sys
    try:
        SESSION_ID=sys.argv[1]
        file_output_dir=sys.argv[2]
        call_pyradiomics(SESSION_ID,file_output_dir)
        print("I AM HERE")
    except Exception as e:
        print(f"I FAILED::{e}")

    #
    # # Ensure input arguments are provided
    # if len(sys.argv) != 3:
    #     print("Usage: python run_pyradiomics.py <grayscale_image> <mask_image>")
    #     sys.exit(1)
    #
    # gray_image = sys.argv[1]
    # mask_image = sys.argv[2]
    #
    # try:
    #     output_file = extract_radiomics_features(gray_image, mask_image)
    #     print(f"Radiomics features saved to {output_file}")
    # except Exception as e:
    #     print(str(e))
    #     sys.exit(1)
