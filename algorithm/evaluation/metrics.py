#==============================================================================
# ###########################################################
#                                                  
#  The following file is used to produce confusion matrices
# 
# ###########################################################
#==============================================================================


import numpy as np
import xlsxwriter

def confusionMatrix(expected,predicted,ModelType,paths=None,numClass=None):
    
    #check if given number of classes, else work it out
    if numClass is None:
        numClass = len(np.unique(expected))
        #print("Number of classes assumed to be {}".format(numClass))
        
    #create an empty matrix of 0s
    cMatrix = [[0] * numClass for i in range(numClass)]
    
    #go through each line and increase its location in the confusion matrix
    for predLine, expLine in zip(predicted, expected):
        cMatrix[int(expLine)][int(predLine)] += 1
    cMatrix = np.asarray(cMatrix)
    
    #write to excel if given a path directory
    if paths is not None:
        results_path = paths['confusion_matrix']
        workbook = xlsxwriter.Workbook(results_path.replace('confusion_matrix.xlsx', ModelType + '_cMatrix.xlsx'))
        worksheet = workbook.add_worksheet()
        row = 0
        for col, data in enumerate(cMatrix):
            worksheet.write_column(row, col, data)
        workbook.close()
    
    return cMatrix