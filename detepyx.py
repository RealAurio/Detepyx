#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 19:05:17 2023

@author: aurio
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
import matplotlib.collections as clt
import matplotlib.animation as anm
import os
import h5py
import time

import tpx3_analysis as TA

output_folder = "../output"

default_color_palette = {"Small Blob": "g", "Medium Blob": "y", "Heavy Blob": "r", 
                         "Straight Track": "b", "Heavy Track": "m", "Light Track": "orange"}

link_direction = {0: (1, 0), 1: (1, 1), 2: (0, 1), 3: (-1, 1), 4: (-1, 0), 5: (-1, -1), 6: (0, -1), 7: (1, -1)}
rotation_ring = {0: (0, 0), 1: (-1, 0), 2: (-1, -1), 3: (0, -1)}
displacement = {2: (0, 0), 4: (-1, 0), 6: (-1, -1), 0: (0, -1)}
index_projection = {0: (0, -1, 0, 0), 1: (0, 0, 0, +1), 2: (+1, 0, 0, 0), 3: (0, 0, +1, 0),
                    4: (0, +1, 0, 0), 5: (0, 0, 0, -1), 6: (-1, 0, 0, 0), 7: (0, 0, -1, 0)}
neighbour_pixels = ((-1,-1), (0,-1), (1,-1), (1,0), (1,1), (0,1), (-1,1), (-1,0))

""" First of all, a Detector class is defined. This class is meant to contain all relevant information about the detector
    used, such as its name (prototype_name), the XY dimensions of the detectors array (size), the intrinsic noise value
    that the detector may have (noise_value) and the the output files' extension (output_type).
    
    When defining a Detector object, one must only pass its prototype_name, and all the specifications are automatically
    retrieved from a predefined dictionary.
    
    In case your detector is not already taken into account here, or in case you want to change some of its predefined
    specifications, you may run the auxiliary function add_custom_detector(prototype_name, prototype_specifications)
    which is defined right below the class. prototype_specifications must be a list or tuple (a tuple is prefered over
    a list in this case) with the following form (Xpixels, Ypixels, noise_value, output_type). Some examples can be
    clearly seen right below this comment. A custom specification will always have priority over predefined ones.
    
    Once defined a Detector instance, you may change the property prototype_name, and the rest of properties will be
    accordingly updated automatically. """

class Detector:
    
    _prototype_specifications = {"Timepix3_tiff_mode": (256, 256, 0, ".tiff"),
                                 "Timepix3": (256, 256, 0, ".tpx3"),
                                 "Minipix": (256, 256, 0, ".txt"),
                                 "CMOS": (2048, 2048, 23, ".tiff")}
    
    custom_specifications = {}
    
    def __init__(self, prototype_name):
        self.prototype_name = prototype_name
    
    @property
    def prototype_name(self):
        return self._prototype_name
    
    @prototype_name.setter
    def prototype_name(self, prototype_name):
        try:
            if prototype_name in Detector.custom_specifications:
                specifications = Detector.custom_specifications[prototype_name]
            else:
                specifications = Detector._prototype_specifications[prototype_name]
            self._Xpixels = specifications[0]
            self._Ypixels = specifications[1]
            self._noise_value = specifications[2]
            self._output_type = specifications[3]
            self._prototype_name = prototype_name
        except KeyError:
            raise Exception(prototype_name + " detector is not currently defined in this module. You may use the function add_custom_detector(prototype_name, prototype_specifications) if you wish to add " + prototype_name + " detector. Alternatively, you could modify the custom_specifications directly on the Detector class definition.")
    
    @property
    def size(self):
        return (self._Xpixels, self._Ypixels)
    
    @property
    def noise_value(self):
        return self._noise_value
    
    @property
    def output_type(self):
        return self._output_type

def add_custom_detector(prototype_name, prototype_specifications):
    if prototype_name in Detector._prototype_specifications:
        print(prototype_name + " was already pre-defined. Saving given input as a custom detector with priority over the pre-defined ones.")
    if prototype_name in Detector.custom_specifications:
        print(prototype_name + " was already defined as a custom detector. Overwritting previous specifications.")
    Detector.custom_specifications[prototype_name] = prototype_specifications

class Acquisition:
    
    def __init__(self, folder_route, detector_used, threshold = 1):
        self.detector_used = detector_used
        self.folder_route = folder_route
        self.threshold = threshold
    
    @property
    def folder_route(self):
        return self._folder_route
    
    @folder_route.setter
    def folder_route(self, folder_route):
        self._folder_route = os.path.abspath(folder_route)
        self._filenames = self._get_filenames()
    
    @property
    def threshold(self):
        return self._threshold
    
    @threshold.setter
    def threshold(self, threshold):
        self._threshold = threshold
    
    @property
    def detector_used(self):
        return self._detector_used.prototype_name
    
    @detector_used.setter
    def detector_used(self, detector_used):
        self._detector_used = Detector(detector_used)
    
    def draw(self, perimeters = False, convex_hulls = False, minimal_rectangles = False, color_palette = default_color_palette, minimum_active_pixels = 1, save = False, save_as = "full_acquisition.pdf", include_event_ID = False, save_in_folder = output_folder):
        event = Event(self._filenames[0], self.detector_used, threshold = self._threshold)
        
        fig = plt.figure(figsize = (10 * 1.2, 10))
        fig.suptitle(self._folder_route, fontsize = 16)
        axis = fig.add_subplot(111)
        fig.tight_layout()
        fig.subplots_adjust(right = 1 / 1.2)
        
        max_X_value, max_Y_value = self._detector_used.size
        
        axis.set_xlim(0, max_X_value)
        axis.set_ylim(0, max_Y_value)
        
        minor_x_ticks = np.arange(0, max_X_value + 1, max_X_value // 2**8)
        minor_y_ticks = np.arange(0, max_Y_value + 1, max_Y_value // 2**8)
        major_x_ticks = np.arange(0, max_X_value + 1, max_X_value // 2**5)
        major_y_ticks = np.arange(0, max_Y_value + 1, max_Y_value // 2**5)
        axis.set_xticks(minor_x_ticks, minor = True)
        axis.set_yticks(minor_y_ticks, minor = True)
        axis.set_xticks(major_x_ticks)
        axis.set_yticks(major_y_ticks)
        axis.grid(True, which='minor', alpha=0.2)
        axis.grid(True, which='major', alpha=0.4)
        patches = []
        for cluster_type, color in default_color_palette.items():
            patches.append(ptc.Patch(color = color, label = cluster_type))
        axis.legend(handles = patches, loc = "center left", bbox_to_anchor=(1.02, 0.5))
        
        fig.text(0.89, 0.6, "M.A.P = " + str(minimum_active_pixels), bbox=dict(boxstyle = "round", ec = (0.8, 0.8, 0.8), fc = (1., 1., 1.)))
        
        if include_event_ID:
            event_ID = 0
        else:
            event_ID = None
        for f in self._filenames:
            event.filename = f
            event.draw(perimeters = perimeters, convex_hulls = convex_hulls, minimal_rectangles = minimal_rectangles, color_palette = color_palette, axis = axis, minimum_active_pixels = minimum_active_pixels, event_ID = event_ID)
            if include_event_ID:
                event_ID += 1
        
        if save == True:
            _, folder_name = os.path.split(self._folder_route)
            save_folder = save_in_folder + "/" + folder_name
            if not os.path.isdir(save_folder):
                os.makedirs(save_folder)
            fig.savefig(save_folder + "/" + save_as)
    
    def classify_clusters(self, save_in_folder = output_folder, minimum_active_pixels = 1):
        _, folder_name = os.path.split(self._folder_route)
        save_folder = save_in_folder + "/" + folder_name
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        events_output = open(save_folder + "/clusters.txt", "w+")
        events_output.write("   ID   |      Filename      |  nº  |  NP  |  IP  | Length | Width |  L/W  | Density | Cluster Type |\n")
        event = Event(self._filenames[0], self.detector_used, threshold = self._threshold)
        event_ID = 0
        for f in self._filenames:
            event.filename = f
            if event.size >= minimum_active_pixels:
                cluster_ID = 1
                clusters = event.clusters
                for cluster in clusters:
                    if cluster.size >= minimum_active_pixels:
                        events_output.write("{:^8}".format(str(event_ID) + "_" + str(cluster_ID)) + "|" + "{:^20}".format(event.raw_filename) + "|" + "{:^6}".format(str(cluster_ID) + "/" + str(len(clusters))) + "|" + "{:^6}".format(cluster.size) + "|" + "{:^6}".format(cluster.number_of_inner_points) + "|" + "{:^8.2f}".format(cluster.length) + "|" + "{:^7.2f}".format(cluster.width) + "|" + "{:^7.3f}".format(cluster.length_over_width) + "|" + "{:^8.3f}%".format(cluster.density*100) + "|" + "{:^14}".format(cluster.cluster_type) + "|" + "\n")
                    cluster_ID += 1
            event_ID += 1
        events_output.close()
    
    def tpx3_to_hdf5(self, boardLayout, cluster = False):
        for f in self._filenames:
            file_route, _ = os.path.splitext(f)
            _, file_name = os.path.split(file_route)
            hdf_folder_route = self._folder_route + "/hdf5folder"
            if not os.path.isdir(hdf_folder_route):
                os.mkdir(hdf_folder_route)
            hdf_route = hdf_folder_route + "/" + file_name + ".hdf5"
            TA.Converter([f], hdf_route, boardLayout, cluster = cluster)
    
    def clusters_3Dplot(self, save_in_folder = output_folder, minimum_active_pixels = 1):
        _, folder_name = os.path.split(self._folder_route)
        save_folder = save_in_folder + "/" + folder_name
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        event = Event(self._filenames[0], self.detector_used, threshold = self._threshold)
        for f in self._filenames:
            event.filename = f
            if event.size >= minimum_active_pixels:
                event.clusters_3Dplot(save_in_folder = save_folder, minimum_active_pixels = minimum_active_pixels, save = True)
    
    def clusters_time_histogram(self, save_in_folder = output_folder, minimum_active_pixels = 1):
        _, folder_name = os.path.split(self._folder_route)
        save_folder = save_in_folder + "/" + folder_name
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        event = Event(self._filenames[0], self.detector_used, threshold = self._threshold)
        for f in self._filenames:
            event.filename = f
            if event.size >= minimum_active_pixels:
                event.clusters_time_histogram(save_in_folder = save_folder, minimum_active_pixels = minimum_active_pixels)
        
    def _get_filenames(self):
        output = []
        for f in os.listdir(self._folder_route):
            if os.path.isfile(os.path.join(self._folder_route, f)):
                file_route = self._folder_route + "/" + f
                _, file_extension = os.path.splitext(file_route)
                if file_extension == self._detector_used.output_type:
                    output.append(file_route)
        return sorted(output)

class Event:
    
    def __init__(self, filename, detector_used, threshold = 1):
        self.detector_used = detector_used
        self._threshold = threshold
        self.filename = filename
    
    @property
    def filename(self):
        return self._filename
    
    @filename.setter
    def filename(self, filename):
        self._filename = filename
        self._set_active_pixels()
        
        if hasattr(self, "_clusters"):
            del self.clusters
    
    @property
    def raw_filename(self):
        _, file_name = os.path.split(self._filename)
        return file_name
    
    @property
    def threshold(self):
        return self._threshold
    
    @threshold.setter
    def threshold(self, threshold):
        self._threshold = threshold
        self._set_active_pixels()
    
    @property
    def detector_used(self):
        return self._detector_used.prototype_name
    
    @detector_used.setter
    def detector_used(self, detector_used):
        self._detector_used = Detector(detector_used)
    
    @property
    def active_pixels(self):
        return self._active_pixels
    
    @property
    def size(self):
        return self._active_pixels.size
    
    @property
    def clusters(self):
        if not hasattr(self, "_clusters"):
            self._clusters = self._active_pixels.clusters
        return self._clusters
    
    @clusters.deleter
    def clusters(self):
        del self._clusters
    
    def _set_active_pixels(self):
        if ".tpx3" == self._detector_used.output_type:
            pixels, values = self._get_data_from_tpx3()
            values_labels = ("toa", "tot", "cluster_index")
            self._active_pixels = ValuedPixelSet(pixels, values, values_labels)
            # self._active_pixels.sort_by_value(0) # 0 -> ToA, so this is sorting by activation time.
        elif ".tiff" == self._detector_used.output_type:
            pixels = self._get_pixels_from_tiff()
            self._active_pixels = PixelSet(pixels)
        elif ".txt" == self._detector_used.output_type:
            pixels = self._get_pixels_from_txt()
            self._active_pixels = PixelSet(pixels)
    
    def _get_pixels_from_txt(self):
        file = open(self._filename, "r")
        lines = file.read().splitlines()
        pixel_list = []
        for i, line in enumerate(lines):
            splitted_line = line.split(" ")
            for j, value in enumerate(splitted_line):
                if int(value.strip()) >= self._threshold:
                    pixel_list.append([i, j])
        file.close()
        output = np.array(pixel_list)
        return output
    
    def _get_pixels_from_tiff(self):
        tiff_image = cv2.imread(self._filename, cv2.IMREAD_GRAYSCALE)
        output = np.transpose((tiff_image >= self._threshold).nonzero())
        return output
    
    def _get_data_from_tpx3(self):
        file_route, _ = os.path.splitext(self._filename)
        acquisition_folder, file_name = os.path.split(file_route)
        hdf_folder_route = acquisition_folder + "/hdf5folder"
        if not os.path.isdir(hdf_folder_route):
            os.mkdir(hdf_folder_route)
        hdf_route = hdf_folder_route + "/" + file_name + ".hdf5"
        if not os.path.isfile(hdf_route):
            TA.Converter([self._filename], hdf_route, "quad", gaps = 1, cluster = True)
            time.sleep(0.01)
        file = h5py.File(hdf_route, "r")
        x = file["x"][:] - 258
        y = file["y"][:] - 258
        toa = file["toa"][:]
        tot = file["tot"][:]
        cluster_index = file["cluster_index"][:]
        file.close()
        output = np.stack((x, y), axis = 1)
        output = output.astype("int16") # Changing data type from uint16 to int16 is needed to prevent overflow on the clusterization algorithm.
        values = np.stack((toa, tot, cluster_index), axis = 1)
        return output, values
    
    def draw(self, perimeters = False, convex_hulls = False, minimal_rectangles = False, color_palette = default_color_palette, axis = None, minimum_active_pixels = 1, save = False, save_as = "event.pdf", event_ID = None, save_in_folder = output_folder):
        if axis == None:
            fig = plt.figure(figsize = (10 * 1.2, 10))
            fig.suptitle(self._filename, fontsize=16)
            axis = fig.add_subplot(111)
            fig.tight_layout()
            fig.subplots_adjust(right = 1 / 1.2)
            
            max_X_value, max_Y_value = self._detector_used.size
            
            axis.set_xlim(0, max_X_value)
            axis.set_ylim(0, max_X_value)
            
            minor_x_ticks = np.arange(0, max_X_value + 1, max_X_value // 2**8)
            minor_y_ticks = np.arange(0, max_Y_value + 1, max_Y_value // 2**8)
            major_x_ticks = np.arange(0, max_X_value + 1, max_X_value // 2**5)
            major_y_ticks = np.arange(0, max_Y_value + 1, max_Y_value // 2**5)
            axis.set_xticks(minor_x_ticks, minor = True)
            axis.set_yticks(minor_y_ticks, minor = True)
            axis.set_xticks(major_x_ticks)
            axis.set_yticks(major_y_ticks)
            axis.grid(True, which='minor', alpha=0.2)
            axis.grid(True, which='major', alpha=0.4)
            patches = []
            for cluster_type, color in default_color_palette.items():
                patches.append(ptc.Patch(color = color, label = cluster_type))
            axis.legend(handles = patches, loc = "center left", bbox_to_anchor=(1.02, 0.5))
            
            fig.text(0.89, 0.6, "M.A.P = " + str(minimum_active_pixels), bbox=dict(boxstyle = "round", ec = (0.8, 0.8, 0.8), fc = (1., 1., 1.)))
        
        if event_ID:
            cluster_ID = 1
            
        for cluster in self._active_pixels.clusters:
            if cluster.size >= minimum_active_pixels:
                cluster.draw(perimeter = perimeters, origin = cluster.origin, axis = axis, convex_hull = convex_hulls, minimal_rectangle = minimal_rectangles, color_palette = color_palette)
                if event_ID:
                    origin = cluster.origin
                    axis.text(origin[0], origin[1], s = str(event_ID) + "_" + str(cluster_ID), fontsize = "small")
                    cluster_ID += 1
        
        if save == True:
            file_route, _ = os.path.splitext(self._filename)
            _, file_name = os.path.split(file_route)
            save_folder = save_in_folder + "/" + file_name
            if not os.path.isdir(save_folder):
                os.makedirs(save_folder)
            
            fig.savefig(save_folder + "/" + save_as)
    
    def timeline(self, save_as, color_palette = default_color_palette):
        
        self._set_artist_iterable(color_palette = color_palette)
        
        fig = plt.figure(figsize = (10, 10))
        fig.suptitle(self._filename, fontsize=16)
        axis = fig.add_subplot(111)
        fig.tight_layout()
        
        max_X_value, max_Y_value = self._detector_used.size
        
        axis.set_xlim(0, max_X_value)
        axis.set_ylim(0, max_X_value)
        
        minor_x_ticks = np.arange(0, max_X_value + 1, max_X_value // 2**8)
        minor_y_ticks = np.arange(0, max_Y_value + 1, max_Y_value // 2**8)
        major_x_ticks = np.arange(0, max_X_value + 1, max_X_value // 2**5)
        major_y_ticks = np.arange(0, max_Y_value + 1, max_Y_value // 2**5)
        axis.set_xticks(minor_x_ticks, minor = True)
        axis.set_yticks(minor_y_ticks, minor = True)
        axis.set_xticks(major_x_ticks)
        axis.set_yticks(major_y_ticks)
        axis.grid(True, which='minor', alpha=0.2)
        axis.grid(True, which='major', alpha=0.4)
        
        def init():
            return []
    
        def animate(i):
            patches = []
            for j in range(i):
                patches.append(axis.add_patch(self._patches[j]))
            return patches
        
        anim = anm.FuncAnimation(fig, animate, init_func = init, frames = len(self._patches), interval = 50, blit = True)
        # writervideo = anm.FFMpegWriter(fps = 60) 
        anim.save(save_as)
    
    def clusters_3Dplot(self, save_in_folder = output_folder, minimum_active_pixels = 1, save = False):
        if self.size >= minimum_active_pixels:
            file_route, _ = os.path.splitext(self._filename)
            _, file_name = os.path.split(file_route)
            save_folder = save_in_folder + "/" + file_name
            if not os.path.isdir(save_folder):
                os.makedirs(save_folder)
            for i, cluster in enumerate(self.clusters):
                if cluster.size >= minimum_active_pixels:
                    if save:
                        cluster.value_surface(0, 1, 2, save_as = save_folder + "/" + str(i) + ".png", relative_to_minimum = True)
                    else:
                        cluster.value_surface(0, 1, 2, relative_to_minimum = True)
        else:
            print("Ommiting " + self._filename + " since it does not contain enough pixels.")
    
    def clusters_time_histogram(self, save_in_folder = output_folder, minimum_active_pixels = 1):
        if self.size >= minimum_active_pixels:
            file_route, _ = os.path.splitext(self._filename)
            _, file_name = os.path.split(file_route)
            save_folder = save_in_folder + "/" + file_name
            if not os.path.isdir(save_folder):
                os.makedirs(save_folder)
            for i, cluster in enumerate(self.clusters):
                if cluster.size >= minimum_active_pixels:
                    cluster.value_histogram(0, save_as = save_folder + "/time_histogram_" + str(i) + ".png", bins = cluster.size//3)
        else:
            print("Ommiting " + self._filename + " since it does not contain enough pixels.")
    
    def _set_artist_iterable(self, color_palette = default_color_palette):
        self._patches = []
        for pixel in self._active_pixels.pixels:
            for cluster in self._active_pixels.clusters:
                if (pixel == cluster.pixels_zoomed + cluster.origin).all(1).any():
                    color = color_palette[cluster.cluster_type]
            self._patches.append(ptc.Rectangle(pixel, 1, 1, facecolor = color))

""" As the name suggests, a PixelSet class is defined as a set of pixels. In order to initialize an instance of this class,
    one must provide an ndarray containing the pixel indexes on each row.
    
    The list of properties the user may access is the following:
        
        + 'pixels': A ndarray containing the pixels in the set. You may change the pixels defining the set providing a new
        ndarray of pixels (this property has a setter).
        
        + 'size': The number of pixels in the set.
        
        + 'clusters': A list of Cluster objects that constitute the pixel set.
    
    This class does not have any public methods. """

class PixelSet:
    
    def __init__(self, pixels):
        self.pixels = pixels
    
    @property
    def pixels(self):
        return self._pixels
    
    @pixels.setter
    def pixels(self, pixels):
        self._pixels = pixels
        
        if hasattr(self, "_clusters"):
            del self.clusters
    
    @property
    def size(self):
        return len(self._pixels)
    
    @property
    def clusters(self):
        if not hasattr(self, "_clusters"):
            self._clusters = self._clusterize()
        return self._clusters
    
    @clusters.deleter
    def clusters(self):
        del self._clusters
    
    def _clusterize(self):
        pixels = np.copy(self._pixels)
        clusters = ()
        while len(pixels) != 0:
            buffer = [pixels[0]]
            current_cluster_pixels = np.array([pixels[0]])
            pixels = np.delete(pixels, 0, axis = 0)
            while len(buffer) != 0:
                search_center = buffer.pop(0)
                for possible_neighbour in neighbour_pixels:
                    neighbour_pixel_locator = (pixels - search_center == possible_neighbour).all(1)
                    if any(neighbour_pixel_locator):
                        neighbour_pixel = pixels[neighbour_pixel_locator]
                        current_cluster_pixels = np.insert(current_cluster_pixels, 0, neighbour_pixel, axis = 0)
                        buffer.append(neighbour_pixel)
                        pixels = np.delete(pixels, neighbour_pixel_locator, axis = 0)
            clusters = clusters + (Cluster(current_cluster_pixels),)
        return clusters

""" In addition to the PixelSet class, a ValuedPixelSet class is also included in this code. The purpose of this class is,
    to handle the cases in whose the user may need to assign a set of values to each pixel in the set. This class thus
    inherits all attributes, methods and properties from the PixelSet class, and adds the following.
    
    In regards to the additional properties:
        
        + 'values': A ndarray with the same rows as pixels, and with as many columns as values are assigned to each pixel.
        The i-th row of this ndarray is assigned to the i-th pixel of the set. You may change this ndarray providing a new
        one (this property disposes of a setter).
        
        + 'values_labels': A tuple of strings containing the name assigned to each value type in the values ndarray. It
        should have as many elements as columns in values. You may change this tuple providing a new one (this porperty has
        a setter).
    
    When it comes down to the public methods, these are the following:
        
        + 'sort_by_value()': This method allows the user to rearrange the order of the pixels and values in the set such
        that a certain value column is ordered in increasing magnitude.
            I) Parameters:
                - 'value_index': Index of the column value to be sorted.
            II) Return: None.
        
        + 'value_surface()': This method makes a 3Dplot of the XY coordinates with a certain specified value in the Z axis.
            I) Parameters:
                - 'value_index': Index of the column value to be plotted in the Z axis.
                
                - 'size_indexer': Index of the column value to be used to weight the size of each point in the plot.
                
                - 'color_indexer': Index of the column value to be used to control the color of each point in the plot.
            II) Return: None. """

class ValuedPixelSet(PixelSet):
    
    """ TO DO: """
    """ +THE FUNCTION value_surface() MUST BE RETHOUGHT, CURRENT VERSION IS TEMPORAL """
    
    def __init__(self, pixels, values, values_labels):
        super().__init__(pixels)
        self.values = values
        self.values_labels = values_labels
    
    @property
    def values(self):
        return self._values
    
    @values.setter
    def values(self, values):
        self._values = values
    
    @property
    def values_labels(self):
        return self._values_labels
    
    @values_labels.setter
    def values_labels(self, values_labels):
        self._values_labels = values_labels
    
    def sort_by_value(self, value_index):
        sorter = np.argsort(self._values[:, value_index])
        self._pixels = self._pixels[sorter]
        self._values = self._values[sorter]
    
    def value_surface(self, value_index, size_indexer, color_indexer):
        color_list=["r", "b", "orange", "m"]
        fig = plt.figure()
        axis = fig.add_subplot(projection='3d')
        X = self._pixels[:,0]
        Y = self._pixels[:,1]
        Z = self._values[:,value_index]
        C = self._values[:,color_indexer]
        S = self._values[:,size_indexer]/min(self._values[:,size_indexer])
        colors = []
        for c in C:
            colors.append(color_list[c])
        axis.scatter(X, Y, Z, marker = 'o', c = colors, s = S)
    
    def _clusterize(self):
        pixels = np.copy(self._pixels)
        values = np.copy(self._values)
        clusters = ()
        while len(pixels) != 0:
            buffer = [pixels[0]]
            current_cluster_pixels = np.array([pixels[0]])
            current_cluster_values = np.array([values[0]])
            pixels = np.delete(pixels, 0, axis = 0)
            values = np.delete(values, 0, axis = 0)
            while len(buffer) != 0:
                search_center = buffer.pop(0)
                for possible_neighbour in neighbour_pixels:
                    neighbour_pixel_locator = (pixels - search_center == possible_neighbour).all(1)
                    if any(neighbour_pixel_locator):
                        neighbour_pixel = pixels[neighbour_pixel_locator]
                        neighbour_value = values[neighbour_pixel_locator]
                        current_cluster_pixels = np.insert(current_cluster_pixels, 0, neighbour_pixel, axis = 0)
                        current_cluster_values = np.insert(current_cluster_values, 0, neighbour_value, axis = 0)
                        buffer.append(neighbour_pixel)
                        pixels = np.delete(pixels, neighbour_pixel_locator, axis = 0)
                        values = np.delete(values, neighbour_pixel_locator, axis = 0)
            clusters = clusters + (ValuedCluster(current_cluster_pixels, current_cluster_values, self._values_labels),)
        return clusters

""" A Cluster class is defined as a connected set of pixels. This class is meant to handle the geometrical classification of
    the clusters detected on each event.
    
    In order to initialize an instance of this class, one must provide an ndarray containing the pixel indexes on each row.
    Many attributes are then automatically set, while others will not be obtained yet until they are needed due to the
    computational cost they carry.
    
    The list of properties the user may access is the following:
        
        + 'pixels_zoomed': A ndarray containing the pixels defining the cluster, but with origin at (0, 0). You may change
        the pixels defining the cluster providing a new ndarray with a different cluster, and all attributes will be updated
        accordingly (this property has a setter).
        
        + 'origin': A tuple containing the minimum value of X and the minimum value of Y. In other words, the vector that has
        been substracted to each pixel position in order to get pixels_zoomed.
        
        + 'size': The number of pixels defining the cluster.
        
        + 'deltaX' and 'deltaY': The size in pixels in the X and Y directions.
        
        + 'length' and 'width': The length and width of the cluster obtained using the rotationg capilars algorithm.
        
        + 'number_of_inner_points': Number of inner pixels that the cluster contains. Default is a minimum of 5 neighbours,
        but this value may be changed at the class variable _minimum_neighbours_for_inner_pixel.
        
        + 'density': The number of pixels divided by the area of the rectangle enclosing the cluster which area is minimal,
        obtained using the rotating capilars algorithm already mentioned.
        
        + 'length_over_width': The length to width ratio of the cluster.
        
        + 'cluster_type': Type of cluster using the geometrical classification explained in 'DOSIMETRIC APPLICATIONS OF
        HYBRID PIXEL DETECTORS' by Stuart P. George at page 52.
        
        + 'cluster_perimeter': The perimeter of the cluster as a ChainCurve, another class defined in this module.
        
    A list of public methods is also provided:
        
        + 'draw()': This method allows the user to plot the cluster as a colormap.
            I) Parameters:
                - 'origin': The origin of the plot. Default is set to (0, 0).
                
                - 'axis': A matplotlib Axes object. The plot will be performed in this Axes. Default is set to None, in which
                case a new figure will be created.
                
                - 'perimeter': Default is set to False. If it is set to True, the ChainCurve defining the cluster perimeter
                will also be plotted.
                
                - 'convex_hull': Default is set to False. If it is set to True, the convex hull containing the cluster will
                also be plotted.
                
                - 'minimal_rectangle': Default is set to False. If it is set to True, the rectangle with minimal area
                containing the cluster will also be plotted.
                
                - 'color_palette': A dictionary assigning a color to each cluster type. The default is set to
                default_color_palette, defined at the beginning of this file.
            II) Return: None. """

class Cluster:
    
    """ TO DO: """
    """ +THE DRAW FUNCTION MUST BE FIXED IN ORDER TO TAKE INTO ACCOUNT THE CASES perimeter = False, minimal_rectangle = True """
    """ +OPTIMIZATION MUST BE DONE: _cluster_to_chain_curve() MUST BE STUDIED AGAIN IN CASE IT CAN BE FURTHER OPTIMIZED"""
    """ +READIBILITY COULD PROBABLY BE IMPROVED """
    
    _minimum_neighbours_for_inner_pixel = 5
    
    _minimum_blob_density = 0.5; _minimum_heavy_track_density = 0.3
    _minimum_straight_track_length_over_width = 8; _maximum_straigth_track_width = 3
    _minimum_heavy_track_length_over_width = 1.25
    
    def __init__(self, pixels):
        self.pixels_zoomed = pixels
    
    @property
    def pixels_zoomed(self):
        return self._pixels_zoomed
    
    @pixels_zoomed.setter
    def pixels_zoomed(self, pixels):
        self._minX = min(pixels[:,0])
        self._minY = min(pixels[:,1])
        self._pixels_zoomed = pixels - (self._minX, self._minY)
        self._deltaX = max(pixels[:,0]) - min(pixels[:,0]) + 1
        self._deltaY = max(pixels[:,1]) - min(pixels[:,1]) + 1
        self._number_of_inner_points = self._count_inner_points()
        
        if hasattr(self, "_length"):
            del self.length
            del self.width
        
        if hasattr(self, "_cluster_type"):
            del self.cluster_type
        
        if hasattr(self, "_cluster_perimeter"):
            del self.cluster_perimeter
    
    @property
    def origin(self):
        return (self._minX, self._minY)
    
    @property
    def size(self):
        return len(self._pixels_zoomed)
    
    @property
    def deltaX(self):
        return self._deltaX
    
    @property
    def deltaY(self):
        return self._deltaY
    
    @property
    def number_of_inner_points(self):
        return self._number_of_inner_points
    
    @property
    def length(self):
        if not hasattr(self, "_length"):
            self._set_length_and_width()
        return self._length
    
    @length.deleter
    def length(self):
        del self._length
    
    @property
    def width(self):
        if not hasattr(self, "_width"):
            self._set_length_and_width()
        return self._width
    
    @width.deleter
    def width(self):
        del self._width
    
    @property
    def density(self):
        return self.size / (self.width * self.length)
    
    @property
    def length_over_width(self):
        return self.length / self.width
    
    @property
    def cluster_type(self):
        if not hasattr(self, "_cluster_type"):
            self._set_cluster_type()
        return self._cluster_type
    
    @cluster_type.deleter
    def cluster_type(self):
        del self._cluster_type
    
    @property
    def cluster_perimeter(self):
        if not hasattr(self, "_cluster_perimeter"):
            self._cluster_perimeter = self._cluster_to_chain_curve()
        return self._cluster_perimeter
    
    @cluster_perimeter.deleter
    def cluster_perimeter(self):
        del self._cluster_perimeter
    
    def draw(self, origin = (0, 0), axis = None, perimeter = False,
                                                 convex_hull = False, 
                                                 minimal_rectangle = False, 
                                                 color_palette = default_color_palette):
        
        if axis == None:
            fig = plt.figure()
            axis = fig.add_subplot(111)
            max_dimension = max(max(self._pixels_zoomed[:,0]), max(self._pixels_zoomed[:,1])) + 3
            axis.set_xlim(-2, max_dimension)
            axis.set_ylim(-2, max_dimension)
            axis.grid(True)
        color = color_palette[self.cluster_type]
        patches = []
        for pixel in self._pixels_zoomed:
            # plt.plot(pixel[0] + 0.5 + origin[0], pixel[1] + 0.5 + origin[1], "r", marker = "s", markersize = 1)
            patches.append(ptc.Rectangle(pixel + origin, 1, 1, facecolor = color))
            # rectangle = patches.Rectangle((0, 0), 3, 3, edgecolor='orange', facecolor="green", linewidth=7)
        collection = clt.PatchCollection(patches, match_original = True)
        axis.add_collection(collection)
        if perimeter:
            curve = self.cluster_perimeter
            curve.draw(origin = self._pixels_zoomed[np.where(self._pixels_zoomed[:,0] == 0)][0] + origin, axis = axis, convex_hull = convex_hull, minimal_rectangle = minimal_rectangle)
        elif convex_hull:
            curve = self.cluster_perimeter
            indexes = curve.convex_hull_indexes()
            x = []; y = []
            for i in indexes:
                pos = curve.position(i)
                x.append(pos[0] + self._pixels_zoomed[np.where(self._pixels_zoomed[:,0] == 0)][0][0] + origin[0]); y.append(pos[1] + self._pixels_zoomed[np.where(self._pixels_zoomed[:,0] == 0)][0][1] + origin[1])
            x.append(x[0]); y.append(y[0])
            axis.plot(x, y, "b-")
    
    def _count_inner_points(self):
        number_of_inner_points = 0
        for pixel in self._pixels_zoomed:
            number_of_neighbours = 0
            for possible_neighbour in neighbour_pixels:
                if any((self._pixels_zoomed - pixel == possible_neighbour).all(1)):
                    number_of_neighbours += 1
            if number_of_neighbours >= Cluster._minimum_neighbours_for_inner_pixel:
                number_of_inner_points += 1
        return number_of_inner_points
    
    # def _cluster_to_chain_curve(self):
    #     start_pixel = self._pixels_zoomed[np.where(self._pixels_zoomed[:,0] == 0)][0]
    #     current_pixel = np.copy(start_pixel)
    #     A = []
    #     tangent_link = 2
    #     new_tangent_link = 2
    #     while any(current_pixel != start_pixel) or len(A) == 0:
    #         A.append(tangent_link)
    #         i = 1
    #         moved = False
    #         while not moved:
    #             if any((self._pixels_zoomed == current_pixel + link_direction[(tangent_link + i) % 8] + displacement[new_tangent_link]).all(1)) or i == -1:
    #                 new_tangent_link = (tangent_link + 2 * i) % 8
    #                 moved = True
    #             i -= 1
    #         current_pixel = current_pixel + link_direction[tangent_link]
    #         tangent_link = new_tangent_link
    #     return ChainCurve(A)
    
    def _cluster_to_chain_curve(self):
        start_pixel = self._pixels_zoomed[np.where(self._pixels_zoomed[:,0] == 0)][0]
        current_pixel = np.copy(start_pixel)
        A = []
        tangent_link = 2
        while any(current_pixel != start_pixel) or len(A) == 0:
            A.append(tangent_link)
            i = 1
            while not (any((self._pixels_zoomed == current_pixel + link_direction[(tangent_link + i) % 8] + displacement[tangent_link]).all(1)) or i == -1):
                i -= 1
            current_pixel = current_pixel + link_direction[tangent_link]
            tangent_link = (tangent_link + 2 * i) % 8
        return ChainCurve(A)
    
    def _set_length_and_width(self):
        curve = self.cluster_perimeter
        side1, side2 = curve.minimum_area_encasing_rectangle()[1:3]
        if side1 >= side2:
            self._length = side1
            self._width = side2
        else:
            self._length = side2
            self._width = side1
    
    def _set_cluster_type(self):
        
        self._cluster_type = "Light Track"
        
        if self.number_of_inner_points == 0:
            if 1 <= self.size <= 2 or (self.length_over_width == 1 and 3 <= self.size <= 4):
                self._cluster_type = "Small Blob"
            
            elif self.length_over_width > Cluster._minimum_straight_track_length_over_width and self.width < Cluster._maximum_straigth_track_width:
                self._cluster_type = "Straight Track"
        
        elif self.number_of_inner_points > 4:
            if self.length_over_width > Cluster._minimum_heavy_track_length_over_width and self.density > Cluster._minimum_heavy_track_density:
                self._cluster_type = "Heavy Track"
            
            elif self.length_over_width <= Cluster._minimum_heavy_track_length_over_width and self.density > Cluster._minimum_blob_density:
                self._cluster_type = "Heavy Blob"
        
        elif self.number_of_inner_points >= 1:
            if self.length_over_width <= Cluster._minimum_heavy_track_length_over_width and self.density > Cluster._minimum_blob_density:
                self._cluster_type = "Medium Blob"

""" For the same reason the ValuedPixelSet class was defined, a ValuedCluster class is defined. This class inherits all
    attributes, methods and properties from the Cluster class, and adds the following.
    
    In relation to the added properties, these are the following:
        
        + 'values': A ndarray with the same rows as pixels, and with as many columns as values are assigned to each pixel.
        The i-th row of this ndarray is assigned to the i-th pixel of the set. You may change this ndarray providing a new
        one (this property disposes of a setter).
        
        + 'values_labels': A tuple of strings containing the name assigned to each value type in the values ndarray. It
        should have as many elements as columns in values. You may change this tuple providing a new one (this porperty has
        a setter).
    
    In regards to the new public methods:
        
        + 'sort_by_value()': This method allows the user to rearrange the order of the pixels and values in the cluster such
        that a certain value column is ordered in increasing magnitude.
            I) Parameters:
                - 'value_index': Index of the column value to be sorted.
            II) Return: None.
        
        + 'value_surface()': This method makes a 3Dplot of the XY coordinates with a certain specified value in the Z axis.
            I) Parameters:
                - 'value_index': Index of the column value to be plotted in the Z axis.
                
                - 'size_indexer': Index of the column value to be used to weight the size of each point in the plot.
                
                - 'color_indexer': Index of the column value to be used to control the color of each point in the plot.
                
                - 'save_as': Default is None, in which case the plot is not saved. In other case, the plot is saved with
                the name and extension (and path if this were the case) given by the string save_as.
                
                - 'relative_to_minimum': Default is False. If it were to be set to True, the values to be plotted in the z
                axis would be displaced such that the minimum value lays in z = 0.
            II) Return: None.
        
        + 'value_histohgram()': This method makes a histogram of the absolute frequencies of a given values column.
            I) Parameters:
                - 'value_index': Index of the column value to be histogrammed.
                
                - 'save_as': Default is None, in which case the histogram is not saved. In other case, the histogram is
                saved with the name and extension (and path if this were the case) given by the string save_as.
                
                - 'bins': Default is 10. Number of bins to be used in the histogram.
            II) Return: None. """

class ValuedCluster(Cluster):
    
    """ TO DO: """
    """ +THE FUNCTION value_surface() MUST BE RETHOUGHT, CURRENT VERSION IS TEMPORAL """
    
    def __init__(self, pixels, values, values_labels):
        super().__init__(pixels)
        
        self.values = values
        self.values_labels = values_labels
    
    @property
    def values(self):
        return self._values
    
    @values.setter
    def values(self, values):
        self._values = values
    
    @property
    def values_labels(self):
        return self._values_labels
    
    @values_labels.setter
    def values_labels(self, values_labels):
        self._values_labels = values_labels
    
    def sort_by_value(self, value_index):
        sorter = np.argsort(self._values[:, value_index])
        self._pixels_zoomed = self._pixels_zoomed[sorter]
        self._values = self._values[sorter]
    
    def value_surface(self, value_index, size_indexer, color_indexer, save_as = None, relative_to_minimum = False):
        color_list=["r", "b", "orange", "m", "k", "g", "pink"]
        fig = plt.figure()
        axis = fig.add_subplot(projection='3d')
        X = self._pixels_zoomed[:,0]
        Y = self._pixels_zoomed[:,1]
        Z = self._values[:,value_index]
        if relative_to_minimum:
            Z = Z - np.min(Z)
        C = self._values[:,color_indexer]
        S = self._values[:,size_indexer]/min(self._values[:,size_indexer])
        colors = []
        for c in C:
            colors.append(color_list[c])
        axis.scatter(X, Y, Z, marker = 'o', c = colors, s = S)
        side = max((self._deltaX, self._deltaY))
        axis.set_xlim(0, side)
        axis.set_ylim(0, side)
        axis.set_xlabel(r"x pixels")
        axis.set_ylabel(r"y pixels")
        axis.set_zlabel(self._values_labels[value_index])
        fig.suptitle("Size controlled by " + self._values_labels[size_indexer] + " and color controlled by " + self._values_labels[color_indexer])
        if save_as:
            fig.savefig(save_as)
            plt.close()
    
    def value_histogram(self, value_index, save_as = None, bins = 10):
        fig = plt.figure()
        axis = fig.add_subplot(111)
        
        values = self._values[:,value_index]
        axis.hist(values-np.min(values), bins = bins)
        
        axis.set_xlabel(self._values_labels[value_index])
        axis.set_ylabel(r"Absolute frequency")
        if save_as:
            fig.savefig(save_as)
            plt.close()

""" A ChainCurve class is defined as a concatenation of short line segments with 8 possible directions. This line segments
    are represented with an integer number from 0 to 7 as follows:
        
        + 0 -> [1, 0]
        + 1 -> [1, 1]
        + 2 -> [0, 1]
        + 3 -> [-1, 1]
        + 4 -> [-1, 0]
        + 5 -> [-1, -1]
        + 6 -> [0, -1]
        + 7 -> [1, -1]
    
    In order to initialize an instance of this class, the user must provide a list or tuple of integers that constructs the
    curve, taking into account the relation between the integers from 0 to 7 and their respective segment.
    
    This class has the following properties:
        
        + 'A': A list or tuple containing the integers that define the curve. The user may change this property, since it has
        a setter method.
        
        + 'size': The number of segments that define the curve.
    
    When it comes down to the public methods, these are the following:
        
        + 'position()': This method calculates the position of the endpoint of a specific segment of the curve.
            I) Parameters:
                - 'index': The list index of the segment which position is going to be calculated.
            II) Return: A ndarray representing the position.
        
        + 'WH()': This method calculates all positions. There is some redundance with previous function, so in the near
        future one of these will replace the other one, or they might just dissapear and a positions property may be
        defined.
            I) Parameters: None.
            II) Return: (W, H), being W (H) a list containing all x (y) components of the positions.
        
        + 'convex_hull_indexes()': This method uses the rotating capilars algorithm in order to obtain the indexes of the
        segments whose endpoints define the convex hull of minimal area that encloses the curve.
            I) Parameters: None.
            II) Return: A list with those indexes.
        
        + 'minimum_area_encasing_rectangle()': This method uses the convex hull obtained with the other function and
        obtains the minimal area rectangle that contains the curve, using the rotating capilars algorithm.
            I) Parameters: None.
            II) Return: (V, s1, s2, A), being V a ndarray containing the 4 vertices of the rectangle, s1 and s2 the sides'
            lengths of the rectangle and A the area of the rectangle.
        
        + 'draw()': This method allows the user to draw the curve.
            I) Parameters:
                - 'origin': The origin of the plot. Default is set to (0, 0).
                
                - 'axis': A matplotlib Axes object. The plot will be performed in this Axes. Default is set to None, in which
                case a new figure will be created.
                
                - 'convex_hull': Default is set to False. If it is set to True, the convex hull containing the curve will
                also be plotted.
                
                - 'minimal_rectangle': Default is set to False. If it is set to True, the rectangle with minimal area
                containing the curve will also be plotted.
            II) Return: None. """

class ChainCurve:
    
    """ TO DO: """
    """ +OPTIMIZATION IS YET TO BE DONE. """
    """ +A LOT OF USEFUL METHODS COULD STILL BE ADDED. """
    """ +THERE ARE SOME REDUNDANT METHODS THAT SHOULD BE REPLACED OR REMOVED ALTOGETHER, WHICH MAKES THE CODE A LITTLE BIT
        TOO CONVOLUTED. """
    
    def __init__(self, A):
        self.A = A
    
    @property
    def A(self):
        return self._A
    
    @A.setter
    def A(self, A):
        self._A = A
    
    @property
    def size(self):
        return len(self._A)
    
    def position(self, index):
        pos = np.array([0, 0])
        for i in range(index + 1):
            pos += link_direction[self._A[i]]
        return pos
    
    def draw(self, origin = (0, 0), axis = None, convex_hull = False,
                                                 minimal_rectangle = False):
        if axis == None:
            fig = plt.figure()
            axis = fig.add_subplot(111)
        position = np.copy(origin)
        for a in self._A:
            dr = link_direction[a]
            axis.arrow(position[0], position[1], dr[0], dr[1], head_width = 0.1, length_includes_head = True)
            position = position + dr
        if convex_hull:
            indexes = self.convex_hull_indexes()
            x = []; y = []
            for i in indexes:
                pos = self.position(i)
                x.append(pos[0] + origin[0]); y.append(pos[1] + origin[1])
            x.append(x[0]); y.append(y[0])
            axis.plot(x, y, "b-")
        if minimal_rectangle:
            points = self.minimum_area_encasing_rectangle()[0]
            x = points[:,0] + origin[0]; y = points[:,1] + origin[1]
            x = np.insert(x, 4, x[0]); y = np.insert(y, 4, y[0])
            axis.plot(x, y, "g-")
        return axis
    
    def WH(self):
        W = []; H = []
        s1 = 0; s2 = 0
        for i in range(self.size):
            a = link_direction[self._A[i]]
            s1 += a[0]
            s2 += a[1]
            W.append(s1)
            H.append(s2)
        return W, H
    
    def convex_hull_indexes(self):
        W, H = self.WH()
        wh = [min(W), max(H), max(W), min(H)]
        index_list = []
        current_position = np.array([0, 0])
        corners_reached = 0
        k = 1
        i = 0
        # print("Start")
        #------------------------------IMPROVABLE----------------------------------------------#
        
        while corners_reached != 4:
            current_position += link_direction[self._A[i]]
            if current_position[corners_reached % 2] == wh[corners_reached % 4]:
                if k != 2:
                    index_list.append(i)
                    k += 1
                else:
                    index_list[-1] = i
            if current_position[(corners_reached + 1) % 2] == wh[(corners_reached + 1) % 4]:
                if index_list[-1] != i or len(index_list) == 0:
                    index_list.append(i)
                k = 1
                corners_reached += 1
            i = (i + 1) % self.size
        
        #--------------------------------------------------------------------------------------#
        # print("Step 1")
        
        buffer = []
        for i in range(len(index_list)):
            buffer.append((index_list[i], index_list[(i+1) % len(index_list)]))
        # print("Step 2")
        
        N = len(buffer)
        j = 0
        
        while N != 0:
            pair = buffer.pop(0)
            i_min = pair[0]
            i_max = pair[1]
            # theta = np.arctan2(H[i_max] - H[i_min], W[i_max] - W[i_min])
            # sin = np.sin(theta); cos = np.cos(theta)
            L = np.sqrt((H[i_max] - H[i_min])**2 + (W[i_max] - W[i_min])**2)
            sin = (H[i_max] - H[i_min]) / L ; cos = (W[i_max] - W[i_min]) / L
            # abcd = np.array([cos, sin, cos + sin, cos - sin])
            abcd_projections = np.array([0, 0, 0, 0])
            V = []
            if pair[1] >= pair[0]:
                for i in np.arange(pair[0] + 1, pair[1], 1):
                    abcd_projections += index_projection[self._A[i]]
                    V.append((abcd_projections[0] + abcd_projections[2] + abcd_projections[3]) * cos + (abcd_projections[1] + abcd_projections[2] - abcd_projections[3]) * sin)
            else:
                for i in np.arange(pair[0] + 1, self.size, 1):
                    abcd_projections += index_projection[self._A[i]]
                    V.append((abcd_projections[0] + abcd_projections[2] + abcd_projections[3]) * cos + (abcd_projections[1] + abcd_projections[2] - abcd_projections[3]) * sin)
                for i in np.arange(0, pair[1], 1):
                    abcd_projections += index_projection[self._A[i]]
                    V.append((abcd_projections[0] + abcd_projections[2] + abcd_projections[3]) * cos + (abcd_projections[1] + abcd_projections[2] - abcd_projections[3]) * sin)
            # print(abcd_projections)
            if len(V) != 0:
                if max(V) > 0:
                    max_index = (np.where(V == max(V))[0][0] + pair[0] + 1) % self.size
                    buffer.append((pair[0], max_index))
                    buffer.append((max_index, pair[1]))
                    index_list.append(max_index)
            N = len(buffer)
            j += 1
        return sorted(index_list)
    
    def minimum_area_encasing_rectangle(self):
        convex_hull = self.convex_hull_indexes()
        N = len(convex_hull)
        points = np.zeros(shape = (N, 2))
        
        for i in range(N):
            points[i] = self.position(convex_hull[i])
        
        placeholder = np.zeros(shape = (N, 6))
        for i in range(N):
            side = points[(i+1) % N] - points[i]
            L = np.sqrt(side[0]**2 + side[1]**2)
            # theta = np.arctan2(side[1], side[0])
            cos = side[0]/L
            sin = side[1]/L
            rotation_matrix = np.array([[cos, -sin],
                                        [sin, cos]])
            points_prime = points - points[i]
            for j in range(N):
                points_prime[j] = np.dot(points_prime[j], rotation_matrix)
        
            min_y = min(points_prime[:,1])
            min_x = min(points_prime[:,0])
            max_x = max(points_prime[:,0])
            area = -(max_x - min_x) * min_y
            
            placeholder[i] = np.array([sin, cos, min_x, max_x, min_y, area])
        
        min_area = min(placeholder[:,5])
        minimum_index = np.where(placeholder[:,5] == min_area)[0][0]
        sin, cos, min_x, max_x, min_y, area = placeholder[minimum_index]
        rotation_matrix = np.array([[cos, -sin],
                                    [sin, cos]])
        
        pos1 = np.dot(np.array([min_x, 0]), np.transpose(rotation_matrix)) + points[minimum_index]
        pos2 = np.dot(np.array([max_x, 0]), np.transpose(rotation_matrix)) + points[minimum_index]
        pos3 = np.dot(np.array([max_x, min_y]), np.transpose(rotation_matrix)) + points[minimum_index]
        pos4 = np.dot(np.array([min_x, min_y]), np.transpose(rotation_matrix)) + points[minimum_index]
        
        return (np.array([pos1, pos2, pos3, pos4]), max_x - min_x, -min_y, area)