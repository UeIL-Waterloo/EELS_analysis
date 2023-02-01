import os
import hyperspy.api as hs 
import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from lmfit.models import Model

### Using Tkinter to open file dialogs for file selection by the user 
root = tk.Tk()
root.withdraw()

##### MICROSCOPE SELECTION #####
microscope_select = 2 # A variable which can be changed depending on which microscope was used to acquire the data. Input the desired value corresponding to the microscopes listed below.
if microscope_select == 1:
    print("TALOS 200X Selected")
    set_acceleration_voltage = 200 
    set_collection_angle = 23 
    t_over_lamda =  0.400036406988378 # Talos two linescans (within the same cell): 0.400036406988378
elif microscope_select == 2:
    print("Titan 80-300HB Selected")
    set_acceleration_voltage = 300 
    set_collection_angle = 55 
    t_over_lamda = 0.39191442410 # Talos two linescans (within the same cell):: 0.39191442410
################################

##### DATA STORAGE ARRAYS ##### 

### x and y ###
x_data_array = []
y_data_array = []

### Arrays for the LMFIT best fit data ###
thickness_array = [] 
intensity_ratio_array = []

if microscope_select == 2:
     silicon_nitride_line_scan = input("Is the linescan for an empty nanofluidic cell (no liquid)? ") # Prompt the user to determine if the linescan is for an empty or a hydrated nanofluidic cell assembly
     if silicon_nitride_line_scan  == "Y" or silicon_nitride_line_scan == "Yes" or silicon_nitride_line_scan  == "yes" or silicon_nitride_line_scan  == "y":
         silicon_nitride_line_scan = True # A boolean which establishes that the linescan was performed on an empty nanofluidic cell assembly 
     else:
         silicon_nitride_line_scan = False # A boolean which establishes that the linescan was performed on a hydrated nanofluidic cell assembly 
     print("Please open the file for the linescan")
     file_path = filedialog.askopenfilename() # The user will select the .dm4 file which holds the EELS data
     print(file_path)
     ##### This variable and the subsequent function allow for the creation of folders within the path of the data files #####
     path_to_dm4 = file_path.count("/") # Counting the number of sub-folders within the root folder 
     def remove_last(s):
         num_removed = 0 
         while s.count("/") == path_to_dm4:   
             try:
                 s = s[0:-1]
                 if s.count('/') == path_to_dm4:
                     num_removed += 1
                     continue
                 else:
                     return [s,num_removed]
                 
             except ValueError:
                 return ""
     initial_file_path = remove_last(file_path)[0] # Finding the root file path
     linescan_ID = file_path[len(file_path)-remove_last(file_path)[1]:-4] # Finding the name of the file ("linescan ID")
     print(initial_file_path)
     s = hs.load(file_path) # Using the Hyperspy API to open the .dm4 file
     s_array = np.array(s) # Converting the filetype from H5py to a numpy array for easy manipulation
     main_path = str(initial_file_path) + "/" + str(linescan_ID) # Creating a root folder for the linescan
     # Check whether the specified path exists or not
     isExist = os.path.exists(main_path)
    
     if not isExist:
       # Create a new directory because it does not exist 
       os.makedirs(main_path)
    
     path = str(main_path) + "/" + str(linescan_ID) + str("_EELS_linescan_thickness")
    
     isExist = os.path.exists(path)
    
     if not isExist:
       # Create a new directory because it does not exist 
       os.makedirs(path) 
       
     ##### ANALYSIS OF THE SELECTED .dm4 FILE #####
     count = 0 # First we establish a counter as a means to select where we would like to begin data analysis within the linescan 
     with open(str(path)+"/"+"EELS_thickness_calculation_results.txt", "w") as output: # Within the text file that we have created in the newly established path, we will conduct data analysis and save the results
         while count < len(s_array): # While the counter is less then the length of the collected data file, we will perform data analysis, change this parameter to select individual spectra or groups of spectra from a larger dataset
             output.write(str("#####"+str(s_array[count])+ "#####" + "\n")) # A label to easily search the text file which will be generated 
             data_y = s_array[count] # The intensity of the EELS linescan, this will change as we iterate through the while loop with the counter 
             data_x = np.arange(0, len(data_y), 1) # The x-axis is created as a counter of the step within the linescan 
             max_y = np.max(data_y) # Finding the maximum intensity within the spectrum 
             max_y_index = np.where(data_y == max_y)[0][0] # Finding the index in the numpy array where the spectrum is at an intensity maximum 
             # if max_y_index < 100: # This conditional is used if outliers exist at the beginning of the linescan, eg. a false maximum in the EELS spectrum 
             #     max_y = np.max(data_y[100:])
             #     max_y_index = np.where(data_y == max_y)[0][0]
             half_max = max_y/2
             data_x_corrected = []
             data_x_counter = 0
             while data_x_counter < max_y_index:
                 data_x_corrected.append(-max_y_index+data_x_counter)
                 data_x_counter += 1
             data_x_corrected.append(0)
             data_x_counter = 1
             while len(data_x_corrected) < len(data_y):
                 data_x_corrected.append(data_x_counter)
                 data_x_counter += 1 
                 
             dispersion_value = s.original_metadata.ImageList.TagGroup0.ImageTags.EELS_Spectrometer.Dispersion_eVch # Channel is apparently capitalized for the Talos software but not for the Titan software
             data_x_corrected = np.linspace(data_x_corrected[0]*dispersion_value,(len(data_x_corrected)-max_y_index)*dispersion_value-dispersion_value,num=len(data_x_corrected))
             xdata = np.array(data_x_corrected)
             ydata = data_y
             
             ### Since the half-maximum changes quite substantially from each measurement, here we roughly define the half-maximum value on either side of the ZLP, which gives a pseudo FWHM for the construction of the Gaussian function
             def find_nearest(array, value):
                 array = np.asarray(array)
                 idx = (np.abs(array - value)).argmin()
                 return array[idx]
             
             approximate_half_max_left_point = (find_nearest(ydata[max_y_index-100:max_y_index+1], value=half_max)) # Here we confine the selection of the nearest left data point (in relation to the global maximum) to within 100 data points 
             approximate_half_max_right_point = (find_nearest(ydata[max_y_index:max_y_index+100], value=half_max)) # Here we confine the selection of the nearest right data point (in relation to the global maximum) to within 100 data points 
             nearest_left_index = np.where(ydata == approximate_half_max_left_point)[0][0] # Here we define the index which corresponds to the approximate values set above
             nearest_right_index = np.where(ydata == approximate_half_max_right_point)[0][0]
                 
             # Defining the Gaussian function (unnormalized)
             def gaussian(x, a, b, sig):
                 return a*np.exp(-((x-b)**2)/(2*sig**2))
             
             gmod = Model(gaussian)
             ### Fitting the defined Gaussian function above to a selected range of data ###
             result = gmod.fit(ydata[0:max_y_index+35], x=xdata[0:max_y_index+35], a=max_y, b=xdata[max_y_index], sig=xdata[nearest_left_index:nearest_right_index+1].std())
                     
            ##### Data for figure construction #####
             x_data_array.append(data_x_corrected)
             y_data_array.append(ydata)
            ### Select the first plot shown for the linescan
             if count == 0:
                 data_x_plot_0 = data_x_corrected
                 data_y_plot_0 = data_y
                 lmfit_x_plot_0 = xdata
                 lmfit_y_plot_0 = result.best_fit[0:]
                 path_first_plot = str(main_path) + "/" + str(linescan_ID) + str("_Raw_data_first_plot")
                 isExist = os.path.exists(path_first_plot)
                 if not isExist:
                   # Create a new directory because it does not exist 
                   os.makedirs(path_first_plot)
            
            ### Select a spectrum of interest
             elif count == 179:
                 data_x_plot_233 = data_x_corrected
                 data_y_plot_233 = data_y
                 lmfit_x_plot_233 = xdata
                 lmfit_y_plot_233 = result.best_fit[0:]
                 path_233_plot = str(main_path) + "/" + str(linescan_ID) + str("_Raw_data_179_plot")
                 isExist = os.path.exists(path_233_plot)
                 if not isExist:
                   # Create a new directory because it does not exist 
                   os.makedirs(path_233_plot)
                  
            ### Select the last plot shown for the linescan  
             elif count == len(s_array)-1:
                 data_x_plot_1 = data_x_corrected
                 data_y_plot_1 = data_y
                 lmfit_x_plot_1 = xdata
                 lmfit_y_plot_1 = result.best_fit[0:]
                 path_last_plot = str(main_path) + "/" + str(linescan_ID) + str("_Raw_data_last_plot")
                 isExist = os.path.exists(path_last_plot)
                 if not isExist:
                   # Create a new directory because it does not exist 
                   os.makedirs(path_last_plot)
            
            ########################################
                 
             # Compute the areas using the composite Simpson's rule.
             entire_spectrum_area = simps(data_y, dx=dispersion_value)
             EELS_area = simps(ydata[0:max_y_index+35],xdata[0:max_y_index+35],dx=dispersion_value)
                            
             #### Parameters from the lmfit method ####
             
             a = result.params['a'].value # The height of the curve's peak 
             b = result.params['b'].value # The position of the centre of the peak 
             x1 = xdata[0] # Lower bound
             x2 = xdata[len(xdata)-1] # Upper bound
             sig = result.params['sig'].value # The standard deviation
             n_steps = 100000 # Selecting how finely we would like to perform the integration. Many points are added here to clearly show the shape of the Gaussian fit 
             
             Gauss_x = np.linspace(x1,x2,n_steps)
             
             Gaussian_func_vals = []
             
             for x in Gauss_x:
                 Gauss_y = a*np.exp(-((x-b)**2)/(2*sig**2))
                 Gaussian_func_vals.append(Gauss_y)            
             
             #### Data for figure construction #####
             
             analytical_ZLP = np.sqrt(2*np.pi)*a*sig

             ###############################################################
             ### Determining the log-ratio ###
             nat_log_It_I0 = np.log(entire_spectrum_area/EELS_area)
             
             ### Adding the log-ratio to a storage array ###
             intensity_ratio_array.append(nat_log_It_I0)            
             ###############################################################
             
             ### Define the exp. parameters
             E0_keV = set_acceleration_voltage # Talos 200: 200, TitanHB: 300
             F_rel_factor = (1+E0_keV/1022)/(1+E0_keV/511)**2
             ### A variety of definitions exist for Z_eff, choose the desired definition for the calculation ###
             Z_eff_def = 4 
             if Z_eff_def == 1:
                 # Quadratic mean Z_eff
                 Zeff_Si3N4 = np.sqrt(((3/7)*(14)**2)+((4/7)*(7)**2))
                 # Quadratic mean Z_eff
                 Zeff_H2O = np.sqrt(((2/3)*(1)**2)+((1/3)*(8)**2))
             elif Z_eff_def == 2:
                 # Mean Z_eff
                 Zeff_Si3N4 = (3/7)*14+(4/7)*7
                 # Mean Z_eff
                 Zeff_H2O = (2/3)*1+(1/3)*8
             elif Z_eff_def == 3:    
                 # Hine's Z_eff
                 Zeff_Si3N4 = (((3/7)*(14)**3.1)+((4/7)*(7)**3.1))**(1/3.1)
                 # Hine's Z_eff
                 Zeff_H2O = (((2/3)*1**3.1)+((1/3)*(8)**3.1))**(1/3.1)
             elif Z_eff_def == 4:   
                 # Lenz's Z_eff
                 Zeff_Si3N4 = (((3/7)*(14)**1.3)+((4/7)*(7)**1.3))/(((3/7)*(14)**0.3)+((4/7)*(7)**0.3))
                 # Lenz's Z_eff
                 Zeff_H2O = (((2/3)*1**1.3)+((1/3)*(8)**1.3))/(((2/3)*1**0.3)+((1/3)*(8)**0.3))
             Em_Si3N4_eV = (7.6*Zeff_Si3N4**0.36)
             Em_H2O_eV = (7.6*Zeff_H2O**0.36)
             collection_angle_mrad = set_collection_angle # Talos 200: 23 mrad # Titan HB: 55 mrad
             if silicon_nitride_line_scan == True:
                 IMFP = (106*F_rel_factor*E0_keV)/(Em_Si3N4_eV*np.log((2*collection_angle_mrad*E0_keV)/Em_Si3N4_eV))
                 thickness = IMFP*nat_log_It_I0
             else:
                 IMFP = (106*F_rel_factor*E0_keV)/(Em_H2O_eV*np.log((2*collection_angle_mrad*E0_keV)/Em_H2O_eV))
                 thickness = IMFP*(nat_log_It_I0-t_over_lamda)
             print("Thickness: ", thickness)
             thickness_array.append(thickness)
             
             ### Writing important EELS analysis results to a text file ###
             output.write(str("#################### EELS analysis results from fitting: Result " + str(count+1) + " #################### ")+"\n")
             output.write(str(result.fit_report())+"\n")
             output.write(str("Inelastic mean free path (IMFP): " + str(IMFP))+"\n")
             output.write(str("Thickness: " + str(thickness))+"\n")
             output.write(str("nat_log_It_I0 " + str(nat_log_It_I0))+"\n")  
             output.write(str("Analytical solution (Area of the ZLP): " + str(analytical_ZLP))+"\n") 
             output.write(str("Simpson's method (Area of the ZLP: Negative energies to minimum): " + str(EELS_area))+"\n") 
             output.write(str("Total area of the EELS spectrum: " + str(entire_spectrum_area)) + "\n")
             output.write(str("############################################################################ "))
             
             count += 1   
             print (str(100*count/len(s_array)) + " %")   
             
             
     path_1 = str(main_path) + "/" + str(linescan_ID) + str("_LMFIT_linescan_thicknesses")
     # Check whether the specified path exists or not
     isExist = os.path.exists(path_1)
     
     ### Storing text files for post-analysis in another script / Excel
     if not isExist:                                         
       # Create a new directory because it does not exist 
       os.makedirs(path_1)
     
     ### Calculated thickness 
     with open(str(path_1)+"/"+"thickness_results.txt", "w") as output:
         for i in thickness_array:
             output.write(str(i)+"\n") 
     
     ### Calculated intensity ratio
     with open(str(path_1)+"/"+"intensity_results.txt", "w") as output:
         for i in intensity_ratio_array:
             output.write(str(i)+"\n")
                            
     ##### Saving the first and last spectrum for figure construction #####
     with open(str(path_first_plot)+"/"+"Raw_data_x.txt", "w") as output:
         for i in data_x_plot_0:
             output.write(str(i)+"\n")
     with open(str(path_first_plot)+"/"+"Raw_data_y.txt", "w") as output:
         for i in data_y_plot_0:
             output.write(str(i)+"\n")
     ######################################################################
     with open(str(path_233_plot)+"/"+"Raw_data_x.txt", "w") as output:
         for i in data_x_plot_233:
             output.write(str(i)+"\n")
     with open(str(path_233_plot)+"/"+"Raw_data_y.txt", "w") as output:
         for i in data_y_plot_233:
             output.write(str(i)+"\n") 
     ######################################################################        
     with open(str(path_last_plot)+"/"+"Raw_data_x.txt", "w") as output:
         for i in data_x_plot_1:
             output.write(str(i)+"\n")
     with open(str(path_last_plot)+"/"+"Raw_data_y.txt", "w") as output:
         for i in data_y_plot_1:
             output.write(str(i)+"\n")
     ######################################################################
       
##### Talos 200 EELS Analyis #####
else:
    silicon_nitride_line_scan = input("Is the linescan for an empty nanofluidic cell (no liquid)? ")
    if silicon_nitride_line_scan  == "Y" or silicon_nitride_line_scan == "Yes" or silicon_nitride_line_scan  == "yes" or silicon_nitride_line_scan  == "y":
        silicon_nitride_line_scan = True
    else:
        silicon_nitride_line_scan = False
    print("Please open the file for the linescan")
    file_path = filedialog.askopenfilename()
    print(file_path)
    ##### This variable and the subsequent function allow for the creation of folders within the path of the data files #####
    path_to_dm4 = file_path.count("/") # Counting the number of sub-folders within the root folder 
    def remove_last(s):
        num_removed = 0 
        while s.count("/") == path_to_dm4:   
            try:
                s = s[0:-1]
                if s.count('/') == path_to_dm4:
                    num_removed += 1
                    continue
                else:
                    return [s,num_removed]
                
            except ValueError:
                return ""
    initial_file_path = remove_last(file_path)[0] # Finding the root file path
    linescan_ID = file_path[len(file_path)-remove_last(file_path)[1]:-4] # Finding the name of the file ("linescan ID")
    print(initial_file_path)
    s = hs.load(file_path) # Using the Hyperspy API to open the .dm4 file
    s_array = np.array(s[2]) # Converting the filetype from H5py to a numpy array for easy manipulation. The spectra are contained within the second index of the list
    main_path = str(initial_file_path) + "/" + str(linescan_ID) # Creating a root folder for the linescan
    # Check whether the specified path exists or not
    isExist = os.path.exists(main_path)
    
    if not isExist:
      # Create a new directory because it does not exist 
      os.makedirs(main_path)
    
    path = str(main_path) + "/" + str(linescan_ID) + str("_EELS_linescan_thickness")
    
    isExist = os.path.exists(path)
    
    if not isExist:
      # Create a new directory because it does not exist 
      os.makedirs(path)
      
    ##### ANALYSIS OF THE SELECTED .dm4 FILE ##### 
    count = 0 # First we establish a counter as a means to select where we would like to begin data analysis within the linescan 
    with open(str(path)+"/"+"EELS_thickness_calculation_results.txt", "w") as output: # Within the text file that we have created in the newly established path, we will conduct data analysis and save the results
        while count < len(s_array): # While the counter is less then the length of the collected data file, we will perform data analysis, change this parameter to select individual spectra or groups of spectra from a larger dataset
            output.write(str("#####"+str(s_array[count])+ "#####" + "\n")) # A label to easily search the text file which will be generated 
            data_y = s_array[count] # The intensity of the EELS linescan, this will change as we iterate through the while loop with the counter 
            data_x = np.arange(0, len(data_y), 1) # The x-axis is created as a counter of the step within the linescan 
            max_y = np.max(data_y) # Finding the maximum intensity within the spectrum 
            max_y_index = np.where(data_y == max_y)[0][0] # Finding the index in the numpy array where the spectrum is at an intensity maximum
            # if max_y_index < 100: # This conditional is used if outliers exist at the beginning of the linescan, eg. a false maximum in the EELS spectrum 
            #     max_y = np.max(data_y[100:])
            #     max_y_index = np.where(data_y == max_y)[0][0]
            half_max = max_y/2
            
            data_x_corrected = []
            data_x_counter = 0
            while data_x_counter < max_y_index:
                data_x_corrected.append(-max_y_index+data_x_counter)
                data_x_counter += 1
            data_x_corrected.append(0)
            data_x_counter = 1
            while len(data_x_corrected) < len(data_y):
                data_x_corrected.append(data_x_counter)
                data_x_counter += 1 
                
            dispersion_value = s[2].original_metadata.ImageList.TagGroup0.ImageTags.EELS_Spectrometer.Dispersion_eVCh # Channel is apparently capitalized for the Talos software but not for the Titan software
            data_x_corrected = np.linspace(data_x_corrected[0]*dispersion_value,(len(data_x_corrected)-max_y_index)*dispersion_value-dispersion_value,num=len(data_x_corrected))
            xdata = np.array(data_x_corrected)
            ydata = data_y
            
            ### Since the half-maximum changes quite substantially from each measurement, here we roughly define the half-maximum value on either side of the ZLP, which gives a pseudo FWHM for the construction of the Gaussian function
            def find_nearest(array, value):
                array = np.asarray(array)
                idx = (np.abs(array - value)).argmin()
                return array[idx]
            
            approximate_half_max_left_point = (find_nearest(ydata[max_y_index-100:max_y_index], value=half_max)) # Here we confine the selection of the nearest left data point (in relation to the global maximum) to within 100 data points 
            approximate_half_max_right_point = (find_nearest(ydata[max_y_index:max_y_index+100], value=half_max)) # Here we confine the selection of the nearest right data point (in relation to the global maximum) to within 100 data points 
            nearest_left_index = np.where(ydata == approximate_half_max_left_point)[0][0] # Here we define the index which corresponds to the approximate values set above
            nearest_right_index = np.where(ydata == approximate_half_max_right_point)[0][0]
                
            # Defining the Gaussian function (unnormalized)
            def gaussian(x, a, b, sig):
                return a*np.exp(-((x-b)**2)/(2*sig**2))
            
            gmod = Model(gaussian)
            ### Fitting the defined Gaussian function above to a selected range of data ###
            result = gmod.fit(ydata[0:max_y_index+24], x=xdata[0:max_y_index+24], a=max_y, b=xdata[max_y_index], sig=xdata[nearest_left_index:nearest_right_index+1].std())
                                       
            ##### Data for figure construction #####
            x_data_array.append(data_x_corrected)
            y_data_array.append(ydata)
            ### Select the first plot shown for the linescan
            if count == 0:
                data_x_plot_0 = data_x_corrected
                data_y_plot_0 = data_y
                path_first_plot = str(main_path) + "/" + str(linescan_ID) + str("_Raw_data_first_plot")
                isExist = os.path.exists(path_first_plot)
                if not isExist:
                  # Create a new directory because it does not exist 
                  os.makedirs(path_first_plot)
            
            ### Select a spectrum of interest
            elif count == 179:
                data_x_plot_233 = data_x_corrected
                data_y_plot_233 = data_y
                path_233_plot = str(main_path) + "/" + str(linescan_ID) + str("_Raw_data_179_plot")
                isExist = os.path.exists(path_233_plot)
                if not isExist:
                  # Create a new directory because it does not exist 
                  os.makedirs(path_233_plot)
                  
            ### Select the last plot shown for the linescan  
            elif count == len(s_array)-1:
                data_x_plot_1 = data_x_corrected
                data_y_plot_1 = data_y
                path_last_plot = str(main_path) + "/" + str(linescan_ID) + str("_Raw_data_last_plot")
                isExist = os.path.exists(path_last_plot)
                if not isExist:
                  # Create a new directory because it does not exist 
                  os.makedirs(path_last_plot)
            
            ########################################
                
            # Compute the area using the composite Simpson's rule.
            entire_spectrum_area = simps(data_y, dx=dispersion_value)
            EELS_area = simps(ydata[0:max_y_index+24],xdata[0:max_y_index+24],dx=dispersion_value)
            
            #### Parameters from the lmfit method ####
            
            a = result.params['a'].value # The height of the curve's peak 
            b = result.params['b'].value # The position of the centre of the peak 
            x1 = xdata[0] # Lower bound
            x2 = xdata[len(xdata)-1] # Upper bound
            sig = result.params['sig'].value # The standard deviation
            n_steps = 100000 # Selecting how finely we would like to perform the integration, many points are added here to clearly show the shape of the Gaussian fit 

            Gauss_x = np.linspace(x1,x2,n_steps)
            
            Gaussian_func_vals = []
            
            for x in Gauss_x:
                Gauss_y = a*np.exp(-((x-b)**2)/(2*sig**2))
                Gaussian_func_vals.append(Gauss_y)            
            
            #### Data for figure construction #####
            
            analytical_ZLP = np.sqrt(2*np.pi)*a*sig

            ###############################################################
            ### Determining the log-ratio ###
            nat_log_It_I0 = np.log(entire_spectrum_area/EELS_area)
            
            ### Adding the log-ratio to a storage array ###
            intensity_ratio_array.append(nat_log_It_I0)            
            ###############################################################
            
            ### Define the exp. parameters
            E0_keV = set_acceleration_voltage # Talos 200: 200, TitanHB: 300
            F_rel_factor = (1+E0_keV/1022)/(1+E0_keV/511)**2
            ### A variety of definitions exist for Z_eff, choose the desired definition for the calculation ###
            Z_eff_def = 4 
            if Z_eff_def == 1:
                # Quadratic mean Z_eff
                Zeff_Si3N4 = np.sqrt(((3/7)*(14)**2)+((4/7)*(7)**2))
                # Quadratic mean Z_eff
                Zeff_H2O = np.sqrt(((2/3)*(1)**2)+((1/3)*(8)**2))
            elif Z_eff_def == 2:
                # Mean Z_eff
                Zeff_Si3N4 = (3/7)*14+(4/7)*7
                # Mean Z_eff
                Zeff_H2O = (2/3)*1+(1/3)*8
            elif Z_eff_def == 3:    
                # Hine's Z_eff
                Zeff_Si3N4 = (((3/7)*(14)**3.1)+((4/7)*(7)**3.1))**(1/3.1)
                # Hine's Z_eff
                Zeff_H2O = (((2/3)*1**3.1)+((1/3)*(8)**3.1))**(1/3.1)
            elif Z_eff_def == 4:   
                # Lenz's Z_eff
                Zeff_Si3N4 = (((3/7)*(14)**1.3)+((4/7)*(7)**1.3))/(((3/7)*(14)**0.3)+((4/7)*(7)**0.3))
                # Lenz's Z_eff
                Zeff_H2O = (((2/3)*1**1.3)+((1/3)*(8)**1.3))/(((2/3)*1**0.3)+((1/3)*(8)**0.3))
            Em_Si3N4_eV = (7.6*Zeff_Si3N4**0.36)
            Em_H2O_eV = (7.6*Zeff_H2O**0.36)
            collection_angle_mrad = set_collection_angle # Talos 200: 23 mrad # Titan HB: 55 mrad
            if silicon_nitride_line_scan == True:
                IMFP = (106*F_rel_factor*E0_keV)/(Em_Si3N4_eV*np.log((2*collection_angle_mrad*E0_keV)/Em_Si3N4_eV))
                thickness = IMFP*nat_log_It_I0
            else:
                IMFP = (106*F_rel_factor*E0_keV)/(Em_H2O_eV*np.log((2*collection_angle_mrad*E0_keV)/Em_H2O_eV))
                thickness = IMFP*(nat_log_It_I0-t_over_lamda)
            print("Thickness: ", thickness)
            thickness_array.append(thickness)
                            
            ### Writing important EELS analysis results to a text file ###
            output.write(str("#################### EELS analysis results from fitting: Result " + str(count+1) + " #################### ")+"\n")
            output.write(str(result.fit_report())+"\n")
            output.write(str("Inelastic mean free path (IMFP): " + str(IMFP))+"\n")
            output.write(str("Thickness: " + str(thickness))+"\n") 
            output.write(str("nat_log_It_I0: " + str(nat_log_It_I0))+"\n")  
            output.write(str("Analytical solution (Area of the ZLP): " + str(analytical_ZLP))+"\n") 
            output.write(str("Simpson's method (Area of the ZLP): " + str(EELS_area))+"\n") 
            output.write(str("Total area of the EELS spectrum: " + str(entire_spectrum_area)) + "\n")
            output.write(str("############################################################################ "))
            
            count += 1   
            print (str(100*count/len(s_array)) + " %")
            plt.close("all")
            
            
    path_1 = str(main_path) + "/" + str(linescan_ID) + str("_LMFIT_linescan_thicknesses")
    # Check whether the specified path exists or not
    isExist = os.path.exists(path_1)
    
    ### Storing txt files for post-analysis in another script / Excel
    if not isExist:                                         
      # Create a new directory because it does not exist 
      os.makedirs(path_1)
      
    ### Calculated thickness 
    with open(str(path_1)+"/"+"thickness_results.txt", "w") as output:
        for i in thickness_array:
            output.write(str(i)+"\n") 
            
    ### Calculated intensity ratio
    with open(str(path_1)+"/"+"intensity_results.txt", "w") as output:
        for i in intensity_ratio_array:
            output.write(str(i)+"\n")

    ##### Saving the first and last spectrum for figure construction #####
    with open(str(path_first_plot)+"/"+"Raw_data_x.txt", "w") as output:
        for i in data_x_plot_0:
            output.write(str(i)+"\n")
    with open(str(path_first_plot)+"/"+"Raw_data_y.txt", "w") as output:
        for i in data_y_plot_0:
            output.write(str(i)+"\n")
    ######################################################################
    with open(str(path_233_plot)+"/"+"Raw_data_x.txt", "w") as output:
        for i in data_x_plot_233:
            output.write(str(i)+"\n")
    with open(str(path_233_plot)+"/"+"Raw_data_y.txt", "w") as output:
        for i in data_y_plot_233:
            output.write(str(i)+"\n")
    ######################################################################        
    with open(str(path_last_plot)+"/"+"Raw_data_x.txt", "w") as output:
        for i in data_x_plot_1:
            output.write(str(i)+"\n")
    with open(str(path_last_plot)+"/"+"Raw_data_y.txt", "w") as output:
        for i in data_y_plot_1:
            output.write(str(i)+"\n")
    ######################################################################
    


  




