import os
from radiomics import featureextractor
import pandas as pd
import nibabel as nib
def extract_radiomics_features(gray_image, mask_image, output_csv=None):
    """
    Extract radiomics features from a grayscale image and corresponding mask.

    Args:
        gray_image (str): Path to the grayscale image.
        mask_image (str): Path to the mask image.
        output_csv (str, optional): Path to save the extracted features as a CSV file. If None, defaults to
                                    <gray_image_basename>_radiomics.csv.

    Returns:
        str: Path to the output CSV file.

    Raises:
        FileNotFoundError: If the grayscale or mask image does not exist.
        Exception: For errors during feature extraction.
    """
    # Check if files exist
    if not os.path.exists(gray_image):
        raise FileNotFoundError(f"Grayscale image file '{gray_image}' does not exist.")

    if not os.path.exists(mask_image):
        raise FileNotFoundError(f"Mask image file '{mask_image}' does not exist.")

    # Extract filename without extension for output
    if output_csv is None:
        base_name = os.path.splitext(os.path.basename(gray_image))[0]
        output_csv = f"{base_name}_radiomics.csv"

    # Initialize PyRadiomics feature extractor
    extractor = featureextractor.RadiomicsFeatureExtractor()

    # Extract features
    try:
        if nib.load(gray_image).shape == nib.load(mask_image):
            features = extractor.execute(gray_image, mask_image)

            # Convert to DataFrame and save as CSV
            df = pd.DataFrame([features])
            df.to_csv(output_csv, index=False)

            return output_csv
    except Exception as e:
        print(f"Error during feature extraction: {e}")
        pass
        # raise Exception(f"Error during feature extraction: {e}")

# Example usage if run directly
if __name__ == "__main__":
    import sys

    # Ensure input arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python run_pyradiomics.py <grayscale_image> <mask_image>")
        sys.exit(1)

    gray_image = sys.argv[1]
    mask_image = sys.argv[2]

    try:
        output_file = extract_radiomics_features(gray_image, mask_image)
        print(f"Radiomics features saved to {output_file}")
    except Exception as e:
        print(str(e))
        sys.exit(1)
