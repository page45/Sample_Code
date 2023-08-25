// Creating a region of interest to limit the processing extent
var roi = ee.FeatureCollection('users/patpageERS/Indiana_counties')
  .filter(ee.Filter.eq('COUNTYFP', '097'));

// Cloud filters
function maskS2clouds(image) {
  var qa = image.select('QA60');

  // Bits 10 and 11 are clouds and cirrus, respectively.
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;

  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
      .and(qa.bitwiseAnd(cirrusBitMask).eq(0));

  return image.updateMask(mask).divide(10000);
}

function fmask(image) {
var cloudShadowBitMask = (1 << 4);
var cloudsBitMask = (1 << 3);
var snowBitMask = (1 << 5);
var qa = image.select('QA_PIXEL');
var mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0)
      .and(qa.bitwiseAnd(cloudsBitMask).eq(0))
      .and(qa.bitwiseAnd(snowBitMask).eq(0));
return image.updateMask(mask);
}

// Obtaining sentinel 2 imagery
var sentinel_2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
      .filterDate('2022-04-01', '2022-10-30')
      // Pre-filter to get less cloudy granules.
      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',20))
      .map(maskS2clouds)
      .mean();

// Obtaining Landsat 8 imagery, apply cloud and date filters
var summer2022 = ee.Filter.date('2022-6-21', '2022-9-21');
var landsat_8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
      .select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'ST_B10','QA_PIXEL'])
      .filter(summer2022)
      .map(fmask);

// Applying scaling factors to convert pixel values to SR and ST
function applyScaleFactors(image) {
var opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2);
var thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0);
return image.addBands(opticalBands, null, true)
            .addBands(thermalBands, null, true);}
landsat_8 = landsat_8.map(applyScaleFactors).median();

// Clip the image to the polygon boundary
var landsat_8_clip = landsat_8.clip(roi);
var landsat_8_clip = ee.Image(landsat_8_clip);
                  
// Visualization parameters for thermal
var visParams = {bands: ['ST_B10'],
                  opacity: 1,
                  min: 300,
                  max: 325,
                  palette: ['black', 'blue', 'green', 'yellow', 'orange', 'red', 'white']};

// Creating RGB visualization      
var visRGB = {
  min: 0.0,
  max: 0.3,
  bands: ['B4', 'B3', 'B2'],
};

// Adding additioanl bands to image
var ndvi = sentinel_2.expression('(NIR - Red) / (NIR + Red)', {
  'NIR': sentinel_2.select('B8'),
  'Red': sentinel_2.select('B4')
});

var ndwi = sentinel_2.expression('(Green - NIR) / (Green + NIR)', {
  'NIR': sentinel_2.select('B8'),
  'Green': sentinel_2.select('B4')
});

var savi = sentinel_2.expression('((NIR - Red) / (NIR + Red + 0.5)) * (1.5)', {
  'NIR': sentinel_2.select('B8'),
  'Red': sentinel_2.select('B4')
});

// Creating final Sentinel 2 image
var sentinel_2_image = sentinel_2
  .addBands(ndvi.rename('NDVI'))
  .addBands(ndwi.rename('NDWI'))
  .addBands(savi.rename('SAVI'));

// Creating feature collection of zip code shapefile  
var zipCodes = ee.FeatureCollection('users/patpageERS/Indiana_Zip_Codes');
var MC_zipCodes = zipCodes.filterBounds(roi);
var empty = ee.Image().byte();
var outline_zipCodes = empty.paint({
  featureCollection: zipCodes,
  width: 2
});

// Bands used in classifier
var bands_pred = ['B2', 'B3', 'B4', 'B5', 'B8', 'B9', 'B11', 'NDVI', 'NDWI', 'SAVI'];

// Merging geometries together
var merged_collection = tree_canopy.merge(veget_non_forest).merge(urban).merge(water);

// Creating training and test datasets
var training = sentinel_2_image.select(bands_pred).sampleRegions({
  collection: merged_collection,
  properties: ['id'],
  scale: 20
});

var trainingData = training.randomColumn();
var trainSet = trainingData.filter(ee.Filter.lessThan('random', 0.8));
var testSet = trainingData.filter(ee.Filter.greaterThanOrEquals('random', 0.8));

// Creating random forest classifier
var classifier = ee.Classifier.smileRandomForest(10).train({
  features: trainSet,
  classProperty: 'id',
  inputProperties: bands_pred
});

var classification = sentinel_2_image.select(bands_pred).classify(classifier);

// Centering map and adding map layers
Map.addLayer(sentinel_2_image.clip(roi), visRGB, 'RGB',0);
Map.addLayer(landsat_8_clip, visParams, 'Summer 2022 Surface Temp (K)', 0);
Map.addLayer(classification.clip(roi),{min: 0, max: 3, palette: ['green','yellow', 'grey', 'blue']}, 'Classifier');
Map.addLayer(MC_zipCodes, {color: 'cyan'}, 'Marian County Zip Codes', 0);
Map.addLayer(outline_zipCodes.clip(roi), {palette: 'cyan'}, 'Marian County Zip Code Outlines');
Map.setCenter(-86.1527, 39.7851, 11);

// Assessing accuracy of model
var confusionMatrix = ee.ConfusionMatrix(testSet.classify(classifier)
	.errorMatrix({
		actual: 'id',
		predicted: 'classification'
	}));

// Calculate proportions of tree canopy per zipcode
var roi_class_coverage = ee.Image.pixelArea().addBands(classification)
                      .reduceRegion({
                        reducer: ee.Reducer.sum().group(1),
                        geometry: roi,
                        scale: 10,
                        bestEffort: true
                      });

var zip_class_coverage = ee.Image.pixelArea().addBands(classification)
                      .reduceRegions({
                        reducer: ee.Reducer.sum().group(1),
                        collection: MC_zipCodes,
                        scale: 10
                      });

// Calculate average surface temperature per zipcode
var meanDictionary = landsat_8_clip.reduceRegions({
                      reducer: ee.Reducer.mean(),
                      collection: MC_zipCodes,
                      scale: 30,
});
                      
// Get a download URLs for the FeatureCollection
var downloadUrl_area = zip_class_coverage.getDownloadURL({
  format: 'json',
  selectors: ['geoid20','groups'],
  filename: 'zip_class_coverage'
});

var downloadUrl_temp = meanDictionary.getDownloadURL({
  format: 'json',
  selectors: ['geoid20','ST_B10'],
  filename: 'zip_avg_temp'
});

// Print outputs
print('Confusion Matrix:', confusionMatrix);
print('Overall Accuracy:', confusionMatrix.accuracy());
print('URL for downloading FeatureCollection as JSON (AREA)', downloadUrl_area);
print('URL for downloading FeatureCollection as JSON (TEMP)', downloadUrl_temp);
print('mean dictionary:', meanDictionary);  
print('Class area (m^2) of Marion County:', roi_class_coverage);  
print('Class area (m^2) of each zipcode:', zip_class_coverage.select(['geoid20','groups']));  
// print('Sentinel 2 image stats:', sentinel_2_image);
// print('Training sample polygons:', merged_collection.size());
// print('Training sample size:', training.size());