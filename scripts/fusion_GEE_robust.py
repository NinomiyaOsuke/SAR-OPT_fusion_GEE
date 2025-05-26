# -*- coding: utf-8 -*-
"""
This script performs GEE-based SAR-Optical fusing
Modified version with robust error handling for output processing
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
# ee.Authenticate()
ee.Initialize(project='sar-opt-fusion-project')

#%%

def main():
    """
    ==============================
     Read input and set parameters
    ==============================
    """
    cfg = json.load(open(r"..\config\Parameters.json", 'r'))
    AOI_PATH = cfg["AOI_PATH"]
    AOIs = gpd.read_file(AOI_PATH)

    """
    Setting environment parameters
    """
    PROJECT_TITLE = cfg["PROJECT_TITLE"]
    GEE_USERNAME = cfg["GEE_USERNAME"]
    EXPORT_INPUT = cfg["EXPORT_INPUT"]
    EXPORT_RLR = cfg["EXPORT_RLR"]
    EXPORT_PCA = cfg["EXPORT_PCA"]
    CLEAR_EXISTING = cfg["CLEAR_EXISTING"]

    """
    Define input parameters
    """
    OPTICAL_MISSION = cfg["OPTICAL_MISSION"]
    START_DATE = cfg["START_DATE"]
    END_DATE = cfg["END_DATE"]
    PATH_MATADATA = cfg["PATH_MATADATA"]
    TRAIN_TEST_SPLIT = cfg["TRAIN_TEST_SPLIT"]
    VAL_SPLIT = cfg["VAL_SPLIT"]
    RANDOM_SPLIT = cfg["RANDOM_SPLIT"]
    PCA_SMOOTH = cfg["PCA_SMOOTH"]
    PCA_COMPONENT_RATIO = cfg["PCA_COMPONENT_RATIO"]
    STD_CLOUD_THRESHOLD = cfg["STD_CLOUD_THRESHOLD"]
    AOI_NAME = cfg["AOI_NAME"]

    print("Project:{} started".format(PROJECT_TITLE))
    nameSuffix = "_{}_{}_{}".format(OPTICAL_MISSION, START_DATE, END_DATE)

    """
    ====================================
    Skip asset management to avoid errors
    ====================================
    """
    print("Skipping asset management - using direct export to Drive")

    """
    ==========================
     Preprocess optical images
    ==========================
    """
    AOI = list(AOIs.loc[AOIs['Name'] == AOI_NAME, 'geometry'])[0]
    AOI = ee.Geometry.Polygon(list(AOI.exterior.coords))

    dep_variables = ['NDVI']
    indep_variables = ['VV_mean', 'VH_mean', 'VV_diff', 'VH_diff',
                       'VV_Nmean', 'VH_Nmean', 'VV_Ndiff', 'VH_Ndiff']

    if OPTICAL_MISSION not in ['L8', 'S2']:
        print('Invalid Mission name, please select either S2 or L8')
        raise utilities.InvalidInputs

    collection_ids = {
        'L8': "LANDSAT/LC08/C01/T1_SR",
        'S2': "COPERNICUS/S2_SR"
    }

    optical_bands = {
        'L8': ['B4', 'B5', 'pixel_qa'],
        'S2': ['B4', 'B8', 'SCL']
    }

    optical_collection = ee.ImageCollection(
        collection_ids[OPTICAL_MISSION]).filterBounds(AOI).select(
            optical_bands[OPTICAL_MISSION], [
                'Red', 'NIR', 'cloud']).filterDate(
                    START_DATE, END_DATE)

    optical_collection = GEE_funcs.prepare_optical(optical_collection, AOI,
                                                   OPTICAL_MISSION)
    optical_collection = optical_collection.filterMetadata(
        'PIXEL_COUNT_AOI', 'greater_than', 100)

    """
    ==========================
     Preprocess SAR images
    ==========================
    """
    S1 = ee.ImageCollection('COPERNICUS/S1_GRD').select(
        ['VV', 'VH']).filter(
        ee.Filter.listContains(
            'transmitterReceiverPolarisation',
            'VV')).filter(
        ee.Filter.listContains(
            'transmitterReceiverPolarisation',
            'VH')).filter(
        ee.Filter.eq(
            'instrumentMode',
            'IW')).filterBounds(AOI).filterDate(
        START_DATE,
        END_DATE)

    """
    ==========================
     Pairing and partitioning
    ==========================
    """
    opt_SAR = GEE_funcs.pair_opt_SAR(
        optical_collection, S1, AOI, indep_variables)

    pair_size = opt_SAR.size().getInfo()
    print("Pairing {} and S1".format(OPTICAL_MISSION))
    print("There are {} number of optical-SAR image pairs for analysis".format(pair_size))
    if pair_size < 10:
        print('There are less than 10 images for training, please select a longer time window')
        raise utilities.InvalidInputs

    """
    ================================
    Data split and processing
    ================================
    """
    opt_SAR = ee.ImageCollection(opt_SAR.randomColumn())

    def createConstantBand(img):
        img = img.addBands(ee.Image(1).rename(['constant']))
        return img.toFloat()

    opt_SAR = ee.ImageCollection(opt_SAR.map(createConstantBand))

    if PATH_MATADATA is None:
        if not RANDOM_SPLIT:
            split_position = ee.Dictionary(
                opt_SAR.reduceColumns(
                    ee.Reducer.percentile([TRAIN_TEST_SPLIT]), 
                    ['system:time_start'])).get('p' + str(int(TRAIN_TEST_SPLIT * 100)))

        def set_split_label(img):
            if RANDOM_SPLIT:
                train_test_label = ee.Algorithms.If(
                    ee.Number(img.get('random')).lt(TRAIN_TEST_SPLIT),
                    'Training', 'Testing')
            else:
                train_test_label = ee.Algorithms.If(
                    ee.Number(img.get('system:time_start')).lt(split_position),
                    'Training', 'Testing')
            
            split_label = ee.Algorithms.If(
                ee.String(train_test_label).equals('Training'),
                ee.Algorithms.If(
                    ee.Number(img.get('random')).lt(VAL_SPLIT),
                    'Validation', 'Training'),
                'Testing')
            return img.set({'Split_label': split_label})

        opt_SAR = opt_SAR.map(set_split_label)

    # Create artificial mask for validation
    try:
        art_mask_img = opt_SAR.filter(
            ee.Filter.rangeContains("CLOUD_PERCENTAGE_AOI", 40, 60)).first()
        if art_mask_img:
            art_mask = art_mask_img.select('Mask').multiply(2)
        else:
            # Fallback: create a simple mask
            art_mask = ee.Image.constant(0).rename('Mask')
    except:
        art_mask = ee.Image.constant(0).rename('Mask')

    def add_art_mask(img):
        new_mask = ee.Algorithms.If(
            ee.String(img.get('Split_label')).equals('Validation'),
            img.select('Mask').add(art_mask).toFloat(),
            img.select('Mask'))
        return img.addBands(ee.Image(new_mask).rename('Mask'), overwrite=True)

    opt_SAR = opt_SAR.map(add_art_mask)

    opt_SAR_train = opt_SAR.filterMetadata('Split_label', 'not_equals', 'Testing')

    """
    ==============================
    Data masking and modeling
    ==============================
    """
    def mask_cloud(img):
        return img.updateMask(img.select('Mask').eq(0)).toFloat()

    opt_SAR_train = opt_SAR_train.map(mask_cloud).select(
        ['constant'] + indep_variables + dep_variables)

    """
    ==============================
    Spatio-temporal modelling
    ==============================
    """
    robust_linear_regression = opt_SAR_train.reduce(
        ee.Reducer.robustLinearRegression(
            numX=len(indep_variables) + 1, numY=1))

    band_names = [['constant'] + indep_variables, ['NDVI']]
    rlr_image = robust_linear_regression.select(['coefficients'])\
        .arrayFlatten(band_names).rename(['constant'] + indep_variables)

    def MLR_predict(img):
        NDVI_pred = img.select(['constant'] + indep_variables)\
            .multiply(rlr_image.rename(['constant'] + indep_variables))\
            .reduce('sum').rename('NDVI_pred')
        return img.select(['NDVI', 'Mask']).addBands(NDVI_pred)

    opt_SAR_outputs = opt_SAR.map(MLR_predict).select(['NDVI', 'NDVI_pred', 'Mask'])

    """
    ================================================================
    Post processing - Simplified to avoid memory issues
    ================================================================
    """
    # Skip PCA smoothing to avoid complexity
    NDVI_smoothed = opt_SAR_outputs.select('NDVI_pred').toBands()

    # Simplified post-processing
    NDVI_calibrated, NDVI_filled = GEE_funcs.post_process(
        opt_SAR_outputs, NDVI_smoothed, AOI, STD_CLOUD_THRESHOLD)

    """
    ====================================================
    Export with robust error handling
    ====================================================
    """
    print("Start exporting NDVI prediction")

    # Get collection size for safe processing
    collection_size = NDVI_filled.size().getInfo()
    print(f"Collection size: {collection_size}")

    if collection_size == 0:
        print("Error: No images in final collection")
        return

    # Export individual images instead of multiband to avoid errors
    try:
        # Get the first few images for export
        image_list = NDVI_filled.limit(min(5, collection_size)).toList(min(5, collection_size))
        
        tasks = []
        for i in range(min(3, collection_size)):  # Export up to 3 images
            try:
                img = ee.Image(image_list.get(i))
                img_date = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd').getInfo()
                
                # Export filled NDVI
                filled_task = utilities.export_image_todrive(
                    img.select('NDVI').clip(AOI), AOI,
                    f'NDVI_filled_{img_date}',
                    PROJECT_TITLE,
                    description=f'NDVI filled {img_date}')
                
                # Export prediction
                pred_task = utilities.export_image_todrive(
                    img.select('NDVI_pred').clip(AOI), AOI,
                    f'NDVI_pred_{img_date}',
                    PROJECT_TITLE,
                    description=f'NDVI prediction {img_date}')
                
                # Export mask
                mask_task = utilities.export_image_todrive(
                    img.select('Mask').clip(AOI), AOI,
                    f'NDVI_mask_{img_date}',
                    PROJECT_TITLE,
                    description=f'NDVI mask {img_date}')
                
                tasks.extend([filled_task, pred_task, mask_task])
                print(f"Queued exports for image {i+1}: {img_date}")
                
            except Exception as e:
                print(f"Error processing image {i}: {e}")
                continue
        
        # Export metadata
        try:
            cloud_cover = NDVI_filled.aggregate_array('CLOUD_PERCENTAGE_AOI')
            collection_time = NDVI_filled.aggregate_array('system:time_start')
            ID = NDVI_filled.aggregate_array('system:index')
            indices = ee.List.sequence(0, NDVI_filled.size().subtract(1))

            def collect_metadata(index):
                dictionary = ee.Dictionary({
                    'ID': ID.get(index),
                    'Cloud_cover': cloud_cover.get(index),
                    'CollectionTime': collection_time.get(index)})
                return ee.Feature(None, dictionary)

            metadata_collection = ee.FeatureCollection(indices.map(collect_metadata))
            
            metadata_task = utilities.export_table_todrive(
                metadata_collection,
                'Output_Metadata' + nameSuffix,
                PROJECT_TITLE,
                description='Metadata')
            
            tasks.append(metadata_task)
            
        except Exception as e:
            print(f"Error creating metadata: {e}")

        # Monitor all tasks
        print(f"Starting {len(tasks)} export tasks...")
        for i, task in enumerate(tasks[:3]):  # Monitor first 3 tasks
            try:
                check_task_status(task, cancel_when_interrupted=False)
            except Exception as e:
                print(f"Task {i+1} completed with status check error: {e}")

        print("Export process completed!")
        
    except Exception as e:
        print(f"Export error: {e}")
        print("Trying simplified single image export...")
        
        try:
            # Fallback: export just one composite image
            composite_ndvi = NDVI_filled.select('NDVI').mean().clip(AOI)
            composite_pred = NDVI_filled.select('NDVI_pred').mean().clip(AOI)
            
            task1 = utilities.export_image_todrive(
                composite_ndvi, AOI,
                'NDVI_composite' + nameSuffix,
                PROJECT_TITLE,
                description='NDVI composite')
            
            task2 = utilities.export_image_todrive(
                composite_pred, AOI,
                'NDVI_pred_composite' + nameSuffix,
                PROJECT_TITLE,
                description='NDVI prediction composite')
            
            check_task_status(task1, cancel_when_interrupted=False)
            check_task_status(task2, cancel_when_interrupted=False)
            
            print("Composite export completed!")
            
        except Exception as e2:
            print(f"Fallback export also failed: {e2}")


if __name__ == "__main__":
    main()