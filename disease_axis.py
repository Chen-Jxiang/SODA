# -*- coding: utf-8 -*-
""" disease_axis.py automatically identify the Subtype-Oriented Disease Axes 
(SODA) given the features and multiple labels. It generate the projections of 
the given features (i.e., the disease axes) that optimally separate the 
pairwise comparison betweenthe given labels. 

The package takes two files as the input.

The first file is a file that contains the patient data. The file is separated 
with commas. The format of the file is provided as follows:
    
id,     feature_name1,  feature_name2,  feature_name3,     ... # header
XX0001,     -0.94,          -0.14,          -0.91,         ...
XX0002,     0.77,           0.306           0.86           ...
...

The second file is a file that contains the patient data. The file is separated 
with commas. The labels is represeted using integers. "NA" represents that the 
data is not available. The format of this file is provided as follows:
    
id,         label_name1,    label_name2,    label_name3, ... # header
XX0001,         1,              2,              NA,
XX0002,         0,              2,              0,
...
    

The patients involved in both files are not necessarilly to be the same. The 
package will automatically match the data and the labels based on the patient
id.


To use this package, an example with interactive python console is provided
as follows:

import disease_axis

#Generate the disease axes and save the figures
M = disease_axis.disease_axis("data.csv", "cluster.csv", savefig = True)

M.output_projection("projection.csv") # Output the projection matrix.
M.output_axes("axes.csv") # Output the disease axes.


"""

import numpy as np
from seaborn import violinplot
from sklearn.linear_model import LogisticRegression
import pandas
import matplotlib.pyplot as plt

class disease_axis:
    def __init__(self, filename_features, filename_labels, savefig = False):
        """
        This initializer read two files and generate disease axes.
        
        filename_features is the name of the file that contains the data.
        filename_labels is the name of the file that contains labels.
        savefig is a bool variable, indicating whether the figure generated will
        be automatically saved at the current working directory.
        """
        
        #loading the feature file        
        f_features = np.loadtxt(
            filename_features, str, delimiter = ",")
        self.feature_names = f_features[0]
        self.id_features = f_features[1:, 0]
        self.features = np.array(f_features[1:, 1:], float)
        
        #loading the label file
        f_labels = np.loadtxt(
            filename_labels, str, delimiter = ",")
            
        # Replace "NA" as "-1", such that the whole array can be represented 
        # with integer
        f_labels [f_labels == "NA"] = "-1" 
        
        self.label_names = f_labels[0]
        self.id_labels = f_labels[1:, 0]
        self.labels = np.array(f_labels[1:, 1:], int)
        
        
        #Now we match the features and labels according to the id.
        #We use variable x to represent the features
        #We use variable y to represent the labels
        
        self.id = []
        self.x = []
        self.y = []
        
        for idx_features in range(len(self.id_features)):
            id1 = self.id_features[idx_features]
            if id1 in self.id_labels:
                self.id.append(id1)
                self.x.append(self.features[idx_features])
                idx_labels = np.where(self.id_labels == id1)[0][0]
                self.y.append(self.labels[idx_labels])
        self.x = np.array(self.x)
        self.y = np.array(self.y, int)
        
        #Normalize the data.
        self.x_shift = self.x.mean(0)
        self.x_scale = self.x.std(0)
        self.x = (self.x - self.x_shift) / self.x_scale
        
        
        self.find_axes(savefig)
        
        
        
    def find_axes(self, savefig):
        """
        This function finds the disease axes. 
        
        savefig is a bool variable, indicating whether the figure generated will
        be automatically saved at the current working directory.
        """
        #Initialize the variables.
        self.axes_name = []
        self.axes_coef = []
        self.axes_intercept = []
        self.axes_label1 = []
        self.axes_label2 = []
        
        for iii in range(self.y.shape[1]):
            candidate_labels = np.unique(self.y[:, iii])
            # now we compare each label pairwisely
            for jjj in range(candidate_labels.shape[0]):
                for kkk in range(jjj):
                    label1 = candidate_labels[kkk]
                    label2 = candidate_labels[jjj]
                    if label1 != -1 and label2 != -1:
                        self.axes_name.append( "{}={} vs. {}={}".format(
                            self.label_names[iii + 1], label1, 
                            self.label_names[iii + 1], label2))

                        self.axes_label1.append( [iii, label1] )
                        self.axes_label2.append( [iii, label2] )
                        
                        idx1 = self.y[:, iii] == label1
                        idx2 = self.y[:, iii] == label2
                        
                        idx = np.bitwise_or(idx1, idx2)
                        
                        LR_x = self.x[idx,:]
                        LR_y = self.y[idx, iii]
                        
                        
                        #Find a diasese axis by logistic regression
                        M = LogisticRegression(class_weight = "balanced")
                        M.fit(LR_x, LR_y)
                        
                        self.axes_coef.append(M.coef_)
                        self.axes_intercept.append(M.intercept_)
                        
                        #Plot the figures
                        if savefig:
                            filename = "{}.png".format(
                                self.axes_name[len(self.axes_coef) - 1] )
                        else:
                            filename = None
                        
                        self.plot_violin( 
                            LR_x, LR_y, len(self.axes_coef) - 1, filename)
        plt.show()                    

    
    def plot_violin(self, x, y, axes_idx, filename = None):
        """
        This function generates plot for each disease axis given the data and 
        one pair of the labels
        
        x is the features.
        y is the labels. This function can process binary label only.
        axes_idx is the index for the axis to be plotted
        filename is the file name where the figure will be saved. If filename
        is None, figures will not be saved.
        """
        plt.figure()
        all_labels = np.unique(y)
        assert len(all_labels) == 2 , \
            "This function can process only one pair of labels. Please feed in one pair of the labels。"
        label1 = all_labels[0]
        label2 = all_labels[1]
        
        #Generate the dataframe
        idx1 = y == label1
        idx2 = y == label2
        
        axis = np.dot(self.axes_coef[axes_idx], x.T).flatten()
                          
        
        labels = [[]] * len(axis)
        
        
        for ii in np.where(idx1)[0]:
            labels[ii] = "{} = {}".format(
                self.label_names[self.axes_label1[axes_idx][0] + 1 ], 
                self.axes_label1[axes_idx][1])

        for ii in np.where(idx2)[0]:
            labels[ii] = "{} = {}".format(
                self.label_names[self.axes_label2[axes_idx][0] + 1 ], 
                self.axes_label2[axes_idx][1])
        
                        
        data = pandas.DataFrame()
        
        data["axis"] = axis
        data[self.axes_name[axes_idx] ] = [""] * len(axis)
        data["label"] = labels

        #Generate violin plot based on the generated dataframe.
                                                                                                                                                                                                                                                                                                 
        violinplot(x = "axis", y = self.axes_name[axes_idx],
            hue = "label", data = data, split = True)
            
        plt.plot( [ -self.axes_intercept[axes_idx], -self.axes_intercept[axes_idx] ], 
        [-.4, .4], "k-.", linewidth = 2)
        
        plt.legend(loc = 0)
        
        #save the figure
        if not filename is None:
            plt.savefig(filename)


    def output_projection(self, filename):
        """
        This function output the projection vectors for disease axes to a file.
        
        filename is the name of the file where the results will to outputted.
        """
        f = open(filename, "w")
        f.write( "axes," + ",".join(self.feature_names[1:]) + "\n" )
        for ii in range(len(self.axes_name)):
            f.write(self.axes_name[ii] + ",",)
            f.write( ",".join( np.array( self.axes_coef[ii][0], str ) ))
            f.write("\n")
        f.close()
        
        
        
    def output_axes(self, filename, x = None):
        """
        This function output the disease axes for each patient to a file.
        
        filename is the name of the file where the results will to outputted.
        x is the training data by default, represeted using None. Otherwise,
        x is the data before normalization. The function will altomatically 
        normalize the input features.
        """
        if x is None:
            x = self.x
        else:
            #Normalize the data.
            x = ( x - self.x_shift ) / self.x_scale
        
        #Compute the disease axes.
        axes = np.zeros([len(self.id), len(self.axes_name)])
        
        for ii in range(len(self.axes_name)):
            axes[:, ii] = np.dot(self.axes_coef[ii], x.T)

        #Save to file,
        f = open(filename, "w")
        f.write( "id," + ",".join(self.axes_name) + "\n" )
        for ii in range(len(self.id)):
            f.write(self.id[ii] + ",")
            f.write( ",".join( np.array( axes[ii, :], str ) ))
            f.write("\n")
        f.close()
        
        
        
        
        