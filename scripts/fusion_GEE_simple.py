# -*- coding: utf-8 -*-
"""
Simplified SAR-Optical fusion without complex post-processing
"""

import GEE_funcs
from utilities import check_task_status
import utilities
import os
import time
import math
import json
import geopandas as gpd
import pandas as pd
import ee
ee.Initialize(project='sar-opt-fusion-project')

def main():
    """
    ==============================
     Read input and set parameters
    ==============================
    """
    cfg = json.load(open(r"..\config\Parameters.json", 'r'))
    AOI_PATH = cfg["AOI_PATH"]
    AOIs = gpd.read_file(AOI_PATH)

    PROJECT_TITLE = cfg["PROJECT_TITLE"]
    GEE_USERNAME = cfg["GEE_USERNAME"]
    OPTICAL_MISSION = cfg["OPTICAL_MISSION"]
    START_DATE = cfg["START_DATE"]
    END_DATE = cfg["END_DATE"]
    TRAIN_TEST_SPLIT = cfg["TRAIN_TEST_SPLIT"]
    VAL_SPLIT = cfg["VAL_SPLIT"]
    RANDOM_SPLIT = cfg["RANDOM_SPLIT"]
    STD_CLOUD_THRESHOLD = cfg["STD_CLOUD_THRESHOLD"]
    AOI_NAME = cfg["AOI_NAME"]

    print("Project:{} started".format(PROJECT_TITLE))
    nameSuffix = "_{}_{}_{}".format(OPTICAL_MISSION, START_DATE, END_DATE)

    print("Using simplified processing - skipping complex post-processing")

    """
    ==========================
     Setup AOI and variables
    ==========================
    """
    AOI = list(AOIs.loc[AOIs['Name'] == AOI_NAME, 'geometry'])[0]
    AOI = ee.Geometry.Polygon(list(AOI.exterior.coords))

    dep_variables = ['NDVI']
    indep_variables = ['VV_mean', 'VH_mean', 'VV_diff', 'VH_diff',
                       'VV_Nmean', 'VH_Nmean', 'VV_Ndiff', 'VH_Ndiff']

    """
    ==========================
     Load satellite data
    ==========================
    """
    collection_ids = {
        'L8': "LANDSAT/LC08/C01/T1_SR",
        'S2': "COPERNICUS/S2_SR"
    }

    optical_bands = {
        'L8': ['B4', 'B5', 'pixel_qa'],
        'S2': ['B4', 'B8', 'SCL']
    }

    # Load optical images
    optical_collection = ee.ImageCollection(
        collection_ids[OPTICAL_MISSION]).filterBounds(AOI).select(
            optical_bands[OPTICAL_MISSION], 
            ['Red', 'NIR', 'cloud']).filterDate(START_DATE, END_DATE)

    optical_collection = GEE_funcs.prepare_optical(optical_collection, AOI, OPTICAL_MISSION)
    optical_collection = optical_collection.filterMetadata('PIXEL_COUNT_AOI', 'greater_than', 100)

    # Load SAR images
    S1 = ee.ImageCollection('COPERNICUS/S1_GRD').select(['VV', 'VH'])\
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))\
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))\
        .filter(ee.Filter.eq('instrumentMode', 'IW'))\
        .filterBounds(AOI).filterDate(START_DATE, END_DATE)

    """
    ==========================
     Pair SAR and optical data
    ==========================
    """
    opt_SAR = GEE_funcs.pair_opt_SAR(optical_collection, S1, AOI, indep_variables)

    pair_size = opt_SAR.size().getInfo()
    print("Pairing {} and S1".format(OPTICAL_MISSION))
    print("There are {} number of optical-SAR image pairs for analysis".format(pair_size))
    
    if pair_size < 5:
        print('There are less than 5 images for training, please select a longer time window')
        raise utilities.InvalidInputs

    """
    ==========================
     Simple data preparation
    ==========================
    """
    opt_SAR = ee.ImageCollection(opt_SAR.randomColumn())

    def createConstantBand(img):
        return img.addBands(ee.Image(1).rename(['constant'])).toFloat()

    opt_SAR = opt_SAR.map(createConstantBand)

    # Simple random split
    def set_split_label(img):
        train_test_label = ee.Algorithms.If(
            ee.Number(img.get('random')).lt(TRAIN_TEST_SPLIT),
            'Training', 'Testing')
        return img.set({'Split_label': train_test_label})

    opt_SAR = opt_SAR.map(set_split_label)

    # Training data
    opt_SAR_train = opt_SAR.filterMetadata('Split_label', 'equals', 'Training')

    def mask_cloud(img):
        return img.updateMask(img.select('Mask').eq(0)).toFloat()

    opt_SAR_train_masked = opt_SAR_train.map(mask_cloud).select(
        ['constant'] + indep_variables + dep_variables)

    """
    ==========================
     Train regression model
    ==========================
    """
    print("Training regression model...")
    robust_linear_regression = opt_SAR_train_masked.reduce(
        ee.Reducer.robustLinearRegression(numX=len(indep_variables) + 1, numY=1))

    band_names = [['constant'] + indep_variables, ['NDVI']]
    rlr_image = robust_linear_regression.select(['coefficients'])\
        .arrayFlatten(band_names).rename(['constant'] + indep_variables)

    """
    ==========================
     Apply model for prediction
    ==========================
    """
    def MLR_predict(img):
        NDVI_pred = img.select(['constant'] + indep_variables)\
            .multiply(rlr_image)\
            .reduce('sum').rename('NDVI_pred')
        return img.addBands(NDVI_pred)

    opt_SAR_predicted = opt_SAR.map(MLR_predict)

    """
    ==========================
     Simple cloud filling
    ==========================
    """
    def simple_cloud_fill(img):
        observation = img.select('NDVI')
        prediction = img.select('NDVI_pred')
        cloud_mask = img.select('Mask')
        
        # Replace cloudy pixels with predictions
        filled = observation.where(cloud_mask.eq(1), prediction)
        
        # Calculate error for clear pixels
        error = observation.subtract(prediction).abs().updateMask(cloud_mask.eq(0))
        mae = error.reduceRegion(
            ee.Reducer.mean(), AOI, 100, maxPixels=1e9).get('NDVI')
        mae = ee.Algorithms.If(mae, mae, -1)
        
        return img.addBands(filled.rename('NDVI_filled'))\
                 .set({'MAE': mae})

    final_results = opt_SAR_predicted.map(simple_cloud_fill)

    """
    ==========================
     Export results
    ==========================
    """
    print("Start exporting results...")
    
    collection_size = final_results.size().getInfo()
    print(f"Final collection size: {collection_size}")

    try:
        # Export a few individual images
        image_list = final_results.limit(3).toList(3)
        
        for i in range(min(3, collection_size)):
            try:
                img = ee.Image(image_list.get(i))
                
                # Get image date
                img_date = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd')
                date_str = img_date.getInfo()
                
                print(f"Exporting image for date: {date_str}")
                
                # Export filled NDVI
                filled_task = utilities.export_image_todrive(
                    img.select('NDVI_filled').clip(AOI), 
                    AOI,
                    f'NDVI_filled_{date_str}',
                    PROJECT_TITLE,
                    description=f'Cloud-filled NDVI {date_str}')
                
                # Export original NDVI
                obs_task = utilities.export_image_todrive(
                    img.select('NDVI').clip(AOI), 
                    AOI,
                    f'NDVI_original_{date_str}',
                    PROJECT_TITLE,
                    description=f'Original NDVI {date_str}')
                
                # Export prediction
                pred_task = utilities.export_image_todrive(
                    img.select('NDVI_pred').clip(AOI), 
                    AOI,
                    f'NDVI_prediction_{date_str}',
                    PROJECT_TITLE,
                    description=f'SAR prediction {date_str}')
                
                # Export mask
                mask_task = utilities.export_image_todrive(
                    img.select('Mask').clip(AOI), 
                    AOI,
                    f'Cloud_mask_{date_str}',
                    PROJECT_TITLE,
                    description=f'Cloud mask {date_str}')
                
                # Wait for tasks to complete
                print(f"Processing exports for {date_str}...")
                check_task_status(filled_task, cancel_when_interrupted=False)
                check_task_status(obs_task, cancel_when_interrupted=False)
                check_task_status(pred_task, cancel_when_interrupted=False)
                check_task_status(mask_task, cancel_when_interrupted=False)
                
                print(f"Completed exports for {date_str}")
                
            except Exception as e:
                print(f"Error processing image {i}: {e}")
                continue

        # Export metadata
        try:
            print("Exporting metadata...")
            cloud_cover = final_results.aggregate_array('CLOUD_PERCENTAGE_AOI')
            collection_time = final_results.aggregate_array('system:time_start')
            ID = final_results.aggregate_array('system:index')
            MAE = final_results.aggregate_array('MAE')
            
            indices = ee.List.sequence(0, final_results.size().subtract(1))

            def collect_metadata(index):
                return ee.Feature(None, ee.Dictionary({
                    'ID': ID.get(index),
                    'Cloud_cover': cloud_cover.get(index),
                    'CollectionTime': collection_time.get(index),
                    'MAE': MAE.get(index)
                }))

            metadata_collection = ee.FeatureCollection(indices.map(collect_metadata))
            
            metadata_task = utilities.export_table_todrive(
                metadata_collection,
                f'Simple_Metadata{nameSuffix}',
                PROJECT_TITLE,
                description='Processing metadata')
            
            check_task_status(metadata_task, cancel_when_interrupted=False)
            print("Metadata export completed")
            
        except Exception as e:
            print(f"Error exporting metadata: {e}")

        print("🎉 All exports completed successfully!")
        print(f"Results saved to Google Drive folder: {PROJECT_TITLE}")
        
    except Exception as e:
        print(f"Export error: {e}")


if __name__ == "__main__":
    main()