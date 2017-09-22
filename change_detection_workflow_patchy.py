#******************************************************************************
#  Name: Transformations for change detection and change mask creation
#  Purpose:  Transform pairs of multispectral images into another multispectral image
#            which is a good background to perform change detection

#            -iMAD transforms a pair of images into another multispectral image where each
#                  band is a difference image (the difference of linear combinations of the              
#                   initial images) with maximum variance. Each of these is orthogonal to
#                   the rest.
#
#            -MAF  transforms the input image into a linear combination of its bands where
#                   the spatial autocorrelation (1 pixel horizontal an vertical shift) is
#                    maximized.
#
#             
#
#
#                                  
#
#  Authors: iMAD: Mort Canty optimized by Julian Equihua
#           MAF: Julian Equihua based on the paper by Allan Nielsen
#           Missing data handling: Steffen Gebhardt and Julian Equihua
#           Image thresholding: Julian Equihua


import numpy as np

from geotiffio import readtif
from geotiffio import createtif
from geotiffio import writetif

from scipy import linalg
from scipy import stats
from scipy.stats.distributions import norm
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.filters import median_filter

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from osgeo import gdal

import gc, sys

#from base.logging.function_logger import FunctionLogger
#logger = FunctionLogger(sys.modules[__name__])


def imad(image1, image2, nodatavalue=0,output_file="iMADs.tif"):
    #print "XXX",image1.metadataDict["dataShape"]  
    #cols, rows, bands = image1.metadataDict["dataShape"]    
    #print cols, rows, bands

    # read image 1crop.tif and 2crop.tif which are cropped bitemporal RapidEye images
    dataset1,rows1,cols1,bands1 = readtif(image1)
    dataset2,rows,cols,bands = readtif(image2)

    # save characteristics of the images to be assiged to the output image
    projection = dataset1.GetProjection()
    transform = dataset1.GetGeoTransform()
    driver = dataset1.GetDriver()
      
    #logger.info("Reading input bands")
    # data array (transposed so observations are columns)
    dm = np.zeros((2 * bands, cols * rows))
    k = 0

    for b in xrange(bands):
        band1 = dataset1.GetRasterBand(b+1)
        band1 = band1.ReadAsArray(0, 0, cols, rows).astype(float)
        dm[k, :] = np.ravel(band1)
        
        band2 = dataset2.GetRasterBand(b+1)
        band2 = band2.ReadAsArray(0, 0, cols, rows).astype(float)        
        dm[bands + k, :] = np.ravel(band2)
        k += 1

    # close
    band1 = None
    band2 = None

    #logger.info("Initialize iMAD components")  
    # get no data pixels and extract only valid pixels
    
    nodataidx1 = dm[0, :] == nodatavalue
    nodataidx2 = dm[bands, :] == nodatavalue
    nodataidx = nodataidx1 | nodataidx2
    gooddataidx = nodataidx == False
    dm = dm[:, gooddataidx]
    ngood = np.sum(gooddataidx)
    wt = np.ones(int(ngood))
  
    # iteration of MAD        
    delta = 1.0
    oldrho = np.zeros(bands)
    iter = 0
    print("imad transform running")
    #logger.info("Starting iterations")  
    while (delta > 0.001) and (iter < 50): 
        #print(iter)  
        
        #     weighted covariance matrices and means
        sumw = np.sum(wt)
        means = np.average(dm,axis=1,weights=wt)
        dmc = dm - means[:,np.newaxis]
        dmc = np.multiply(dmc,np.sqrt(wt))
        sigma = np.dot(dmc,dmc.T)/sumw

        #logger.info("sigma")            
        s11 = sigma[0:bands, 0:bands]
        s22 = sigma[bands:, bands:]
        s12 = sigma[0:bands, bands:]
        s21 = sigma[bands:, 0:bands]
        
        #logger.info("solution of generalized eigenproblems")  
        #     solution of generalized eigenproblems
        aux_1 = linalg.solve(s22, s21)
        lama, a = linalg.eig(np.dot(s12, aux_1), s11)
        aux_2 = linalg.solve(s11, s12)
        lamb, b = linalg.eig(np.dot(s21, aux_2), s22)
        
        #logger.info("sorting...")  
        #     sort a  
        idx = np.argsort(lama)
        a = a[:, idx]
        
        #     sort b        
        idx = np.argsort(lamb)
        b = b[:, idx]          
        
        #     canonical correlations        
        rho = np.sqrt(np.real(lamb[idx])) 

        #     normalize dispersions  
        tmp1 = np.dot(np.dot(a.T,s11),a)
        tmp2 = 1. / np.sqrt(np.diag(tmp1))
        tmp3 = np.tile(tmp2, (bands, 1))
        a = np.multiply(a, tmp3)
        b = np.mat(b)
        tmp1 = np.dot(np.dot(b.T,s22),b)
        tmp2 = 1. / np.sqrt(np.diag(tmp1))
        tmp3 = np.tile(tmp2, (bands, 1))
        b = np.multiply(b, tmp3)
        
        #     assure positive correlation
        tmp = np.diag(np.dot(np.dot(a.T,s12),b))
        b = np.dot(b,np.diag(tmp / np.abs(tmp)))

        #print(np.shape(means))
        #     canonical and MAD variates
        U = np.dot(a.T , (dm[0:bands, :] - means[0:bands, np.newaxis]))    
        V = np.dot(b.T , (dm[bands:, :] - means[bands:, np.newaxis]))          
        MAD = U - V  
        
        #     new weights        
        var_mad = np.tile(np.mat(2 * (1 - rho)).T, (1, ngood))    
        chisqr = np.sum(np.multiply(MAD, MAD) / var_mad, 0)
        wt = np.squeeze(1 - np.array(stats.chi2._cdf(chisqr,bands)))
        
        #     continue iteration        
        delta = np.sum(np.abs(rho - oldrho))
        oldrho = rho
        # print delta, rho
        iter += 1

        # reshape to original image size, by including nodata pixels    
    MADout = np.zeros((int(bands + 1), cols * rows))
    MADout[0:bands, gooddataidx] = MAD
    MADout[bands:(bands + 1), gooddataidx] = chisqr

    # close
    MAD = None
    chisqr = None

    if output_file is not None:
        outData = createtif(driver,rows,cols,bands+1,output_file)
        writetif(outData,MADout,projection,transform,order='r')
    else:
        return MADout


def maf(image, no_data_value=0,output_file="iMADs_MAFs.tif"):
    print("maf transform running")

    # this transformation is usually used to postprocess
    # an image resulting from the iMAD transformation.
    # Since these images include de chi-squared statistic
    # it must be discarded:
    image_array,rows,cols,bands = readtif(image)

    # save characteristics of the images to be assiged to the output image
    projection = image_array.GetProjection()
    transform = image_array.GetGeoTransform()
    driver = image_array.GetDriver()

    '''
    Prepare data for maximum autocorrelation factor transform.
    '''
    number_of_maf_variates = bands - 1
    variates_stack = np.zeros((cols * rows, number_of_maf_variates), dtype=float)
    # coavariance matrix from datamodel and itself shifted by 1 pixel
    # (vertically and horizontally)
    rows_1 = rows - 1
    cols_1 = cols - 1
    H = np.zeros((rows_1 * cols_1, number_of_maf_variates), float)
    V = np.zeros((rows_1 * cols_1, number_of_maf_variates), float)
    #logger.info("Reading input bands for MAF transformation")

    for b in range(number_of_maf_variates):
        variates = image_array.GetRasterBand(b + 1)
        variates = variates.ReadAsArray(0, 0, cols, rows).astype(float)
        H[:, b] = np.ravel(variates[0:rows_1, 0:cols_1] - variates[0:rows_1, 1:cols])
        V[:, b] = np.ravel(variates[0:rows_1, 0:cols_1] - variates[1:rows, 0:cols_1])
        variates_stack[:, b] = np.ravel(variates)
    # close useless datamodel data
    image_array = None
    # band_data = None
    # label data considered as NA
    nodataidx = variates_stack[:, :] == no_data_value
    gooddataidx = nodataidx[:, 0] == False
    variates_stack = np.array(variates_stack[gooddataidx, :])
    # variates_stack = np.ma.array(variates_stack, mask=self.no_data_value) #TODO: is this correct?
    # variates_stack = np.ma.masked_values(variates_stack, self.no_data_value) #If this line is executed, the next line (variates_stack.data) erases this execution
    # self.variates_stack = np.array(variates_stack.data[self.gooddataidx, :]) 

    '''
    Perform the maf transformation
    '''
    # covariance of original bands
    # sigma = np.ma.cov(variates_stack.T, allow_masked=True) #TODO: Jenkins can't run this lines with allow_masked =True, whyyyy??
    # covariance for horizontal and vertical shifts
    # sigmadh = np.ma.cov(H.T, allow_masked=True)
    # sigmadv = np.ma.cov(V.T, allow_masked=True)

    sigma = np.cov(variates_stack.T)
    # covariance for horizontal and vertical shifts
    sigmadh = np.cov(H.T)
    sigmadv = np.cov(V.T)

    # simple pooling of shifts
    sigmad = 0.5 * (np.array(sigmadh) + np.array(sigmadv))
    # evalues, vec1 = scipy.linalg.eig(sigmad, sigma)
    evalues, vec1 = linalg.eig(sigmad, sigma)

    # Sort eigen values from smallest to largest and apply this order to
    # eigen vector matrix
    sort_index = evalues.argsort()
    evalues = evalues[sort_index]
    vec1 = vec1[:, sort_index]
    # autocorrelation
    # ac= 1-0.5*vec1
    HH = 1 / np.std(variates_stack, 0, ddof=1)
    diagvariates = np.diag(HH)
    invstderrmaf = np.diag((1 / np.sqrt(np.diag(vec1.T * sigma * vec1))))
    HHH = np.zeros((number_of_maf_variates), float)
    for b in range(number_of_maf_variates):
        # logger.info("Calculating component %d of MAF transformation" % b)  
        HHH[b] = cmp(np.sum((diagvariates * sigma * vec1 * invstderrmaf)[b]), 0)
    sgn = np.diag(HHH)  # assure positiviy
    v = np.dot(vec1, sgn)
    N = np.shape(variates_stack)[0]
    X = variates_stack - np.tile(np.mean(variates_stack, 0), (N, 1))
    # scale v to give MAFs with unit variance
    aux1 = np.dot(np.dot(v.T, sigma), v)  # dispersion of MAFs
    aux2 = 1 / np.sqrt(np.diag(aux1))
    aux3 = np.tile(aux2.T, (number_of_maf_variates, 1))
    v = v * aux3  # now dispersion is unit matrix
    mafs = np.dot(X, v)

    #    reshape to original image size, by including nodata pixels    
    MAFout = np.zeros((cols * rows,number_of_maf_variates))
    MAFout[gooddataidx,:] = mafs

    #   close mafs
    mafs = None

    if output_file is not None:
        outData = createtif(driver,rows,cols,number_of_maf_variates,output_file)
        writetif(outData,MAFout,projection,transform,order='c')
    else:
        return(MAFout)










    
    

