#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 09:45:06 2018

@author: tal
"""

import sys,os,pathlib,re
import glob
import warnings
import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import PIL

import seaborn as sns
import pandas as pd
from tqdm import tqdm
from attrdict import AttrDict

from plotting_utils import get_png_file_info

# map ugly model names to nice model names
model_name_dict={'InceptionV3':'Inception-v3',
'Resnet50':'ResNet-50',
'Resnet_50_l2_eps5':'$\ell_2$-adv-trained (${\epsilon=5}$) ResNet-50',
'Wide_Resnet50_2_l2_eps5':'$\ell_2$-adv-trained (${\epsilon=5}$) WRN-50-2'}


def plot_im_matrix(im_matrix,rows_model,columns_model,c,class_names,subplot_spec=None,panel_label=None):
    n_classes=len(class_names)
    if subplot_spec is None:
        #   start figure
        fig=plt.figure()
        gs0 = gridspec.GridSpec(nrows=3, ncols=3,
                        height_ratios=[c.inch_per_minor_title_space,c.inch_per_image*n_classes,c.minor_margin],
                        width_ratios=[c.inch_per_minor_title_space,c.inch_per_image*n_classes,c.minor_margin],
                        figure=fig,wspace=0,hspace=0,left=0,right=1,bottom=0,top=1)
    else:
        fig=plt.gcf()
        gs0 = gridspec.GridSpecFromSubplotSpec(nrows=3, ncols=3,
                        height_ratios=[c.inch_per_minor_title_space,c.inch_per_image*n_classes,c.minor_margin],
                        width_ratios=[c.inch_per_minor_title_space,c.inch_per_image*n_classes,c.minor_margin],
                        subplot_spec=subplot_spec,wspace=0,hspace=0)

    gs00 = gridspec.GridSpecFromSubplotSpec(nrows=n_classes, ncols=n_classes, subplot_spec=gs0[1,1],hspace=0.0,wspace=0.0)

    for model_1_target_idx, model_1_target in enumerate(class_names):
       for model_2_target_idx, model_2_target in enumerate(class_names):
          cur_im=im_matrix[model_1_target_idx, model_2_target_idx]

          ax = plt.subplot(gs00[model_1_target_idx, model_2_target_idx])

          x_label=model_2_target
          y_label=model_1_target

          if model_1_target_idx==0: # is it the top row?
             ax.set_xlabel(x_label,fontsize=c.class_font_size,labelpad=c.labelpad)
             ax.xaxis.set_label_position('top')

          if model_2_target_idx==0: # is it the leftmost column?
             ax.set_ylabel(y_label,fontsize=c.class_font_size,labelpad=c.labelpad)
             ax.yaxis.set_label_position('left')

          if type(cur_im) is np.ndarray:
                 ax.imshow(cur_im,interpolation='lanczos')
                 #             ax.npz_file=npz_file
#             ax.parent_folder=stimuli_path
#             ax.class_image=cur_im
          elif model_1_target==model_2_target: # diagonal line
              ax.imshow(np.ones([1,1,3]))
              ax.plot([1, 0], [0, 1], 'k-',linewidth=c.line_width,transform=ax.transAxes)
          elif cur_im == 'blank':
              ax.imshow(np.ones([1,1,3]))

              ax.plot([1, 0], [0, 1], 'k-',linewidth=c.line_width,transform=ax.transAxes)
          else: # a missing plot
              # ax.imshow(np.asarray([1.0,0.5,0.5]).reshape([1,1,3]))
              ax.imshow(np.ones([1,1,3]))

              ax.plot([1, 0], [0, 1], 'k-',linewidth=c.line_width,transform=ax.transAxes)
          plt.tick_params(
               axis='x',          # changes apply to the x-axis
               which='both',      # both major and minor ticks are affected
               bottom=False,      # ticks along the bottom edge are off
               top=False,         # ticks along the top edge are off
               labelbottom=False) # labels along the bottom edge are off
          plt.tick_params(
               axis='y',          # changes apply to the x-axis
               which='both',      # both major and minor ticks are affected
               left=False,      # ticks along the bottom edge are off
               right=False,         # ticks along the top edge are off
               labelleft=False)  # labels along the bottom edge are off

          if model_1_target_idx==0 and model_2_target_idx==0 and panel_label is not None:
              plt.text(c.subpanel_letter_x,c.subpanel_letter_y,panel_label,fontsize=c.subpanel_letter_font_size,clip_on=False,fontweight='bold',ha='right',va='bottom')

    x_major_label_ax=plt.Subplot(fig,gs0[0,1])
    fig.add_subplot(x_major_label_ax)
    x_major_label_ax.set_axis_off()
    x_major_label_ax.text(0.5,1.0,model_name_to_title(columns_model),fontdict={'fontsize':c.model_name_font_size},verticalalignment='top',horizontalalignment='center')

    y_major_label_ax=plt.Subplot(fig,gs0[1,0])
    fig.add_subplot(y_major_label_ax)
    y_major_label_ax.set_axis_off()
    y_major_label_ax.text(0.0,0.5,model_name_to_title(rows_model),fontdict={'fontsize':c.model_name_font_size},verticalalignment='center',horizontalalignment='left',rotation=90)

def model_name_to_title(model_name):
    if model_name in model_name_dict.keys():
        return model_name_dict[model_name]
    else:
        return model_name.replace('_',' ')

def nicefy_class_names(class_names):
  return [s.replace('_','\n') for s in class_names]

def plot_single_model_pair_multiple_class_pairs_controversial_stimuli_matrix(subfolder,rows_model,columns_model,c,stimuli_path,subplot_spec=None,panel_label=None):

    # plot a figure like Figure 3 in Golan, Raju & Kriegeskorte, 2020 PNAS
    png_files=glob.glob(os.path.join(stimuli_path,subfolder,'*.png'))
    print("found {} png files.".format(len(png_files)))
    png_files_info=get_png_file_info(png_files)

    print(png_files_info)

    # some sanity checks
    if len(png_files_info)==0:
       warnings.warn('folder {} is empty'.format(subfolder))
       return

    # make sure the folder doesn't have mixed files (i.e., the first model is always one model, and the second is always the other)
    assert len(png_files_info.model_1_name.unique())==1 and len(png_files_info.model_2_name.unique())==1
    model_1_name=png_files_info.model_1_name.unique()[0]
    model_2_name=png_files_info.model_2_name.unique()[0]
    assert ((model_1_name == rows_model) and (model_2_name == columns_model) or
            (model_1_name == columns_model) and (model_2_name == rows_model))

    class_names=list(np.unique(list(png_files_info.model_1_target)+list(png_files_info.model_2_target)))
    n_classes=len(class_names)
    im_matrix=np.empty((n_classes,n_classes), dtype=np.object)

    for png_file,model_1_target,model_2_target in tqdm(zip(png_files_info.filename,png_files_info.model_1_target,png_files_info.model_2_target)):

        cur_im=plt.imread(png_file)

        model_1_target_idx=class_names.index(model_1_target)
        model_2_target_idx=class_names.index(model_2_target)

        if model_1_name==rows_model and model_2_name==columns_model:
            im_matrix[model_1_target_idx, model_2_target_idx]=cur_im
        else:
            im_matrix[model_2_target_idx, model_1_target_idx]=cur_im
    plot_im_matrix(im_matrix,rows_model,columns_model,c,nicefy_class_names(class_names),subplot_spec=subplot_spec,panel_label=panel_label)

def get_subfolders_properties(subfolders):
    list_of_dicts=[]
    import pandas as pd
    for subfolder in subfolders:
        subfolder_m1, subfolder_m2=re.findall(r'([^/]+)_vs_(.+)',subfolder)[0]
        list_of_dicts.append(
                {'name':subfolder,
                 'm1':subfolder_m1,
                 'm2':subfolder_m2,
                 })
    return pd.DataFrame(list_of_dicts)

def plot_8_random_classes_figure(optim_method='direct',target_parent_folder='figures',image_folder='optimization_results'):

    rows_model='Resnet_50_l2_eps5'
    columns_model='Wide_Resnet50_2_l2_eps5'

    n_classes=8

    # figure configuration
    image_width=5.0
    c=AttrDict()
    c.class_font_size=8
    c.labelpad=2.5
    c.model_name_font_size=12
    c.inch_per_minor_title_space=40/72
    c.between_subplot_margin=0/72
    c.minor_margin=5/72
    c.major_margin=5/72
    c.inch_per_major_title_space=15/72
    c.line_width=0.5
    c.inch_per_image=(image_width-c.inch_per_major_title_space-(c.inch_per_minor_title_space+c.minor_margin)*2-c.between_subplot_margin-c.major_margin)/n_classes

    upscale=128
    stimuli_path=os.path.join(image_folder,optim_method+'_optim_8_random_classes')

    plt.close('all')
    fig=plt.figure(figsize=(image_width,image_width))

      # form title/content/margin 3x3 gridspec
    gs0 = gridspec.GridSpec(nrows=3, ncols=3,
                    height_ratios=[c.inch_per_major_title_space,image_width-c.major_margin-c.inch_per_major_title_space,c.major_margin],
                    width_ratios=[c.inch_per_major_title_space,image_width-c.major_margin-c.inch_per_major_title_space,c.major_margin],
                    figure=fig,wspace=0,hspace=0,left=0,right=1,bottom=0,top=1)

    plot_single_model_pair_multiple_class_pairs_controversial_stimuli_matrix('',rows_model=rows_model,columns_model=columns_model,c=c,stimuli_path=stimuli_path,subplot_spec=gs0[1,1])

    fig_fname=optim_method+'_optim_'+rows_model+"_vs_"+columns_model+".pdf"
    pathlib.Path(target_parent_folder).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(target_parent_folder,fig_fname),dpi=upscale/c.inch_per_image)
    plt.savefig(os.path.join(target_parent_folder,fig_fname.replace('.pdf','.png')),dpi=upscale/c.inch_per_image)

    print('saved to',os.path.join(target_parent_folder,fig_fname))


def plot_batch(optimization_methods=None,image_folder='optimization_results',figure_folder='figures'):
  if optimization_methods is None:
    optimization_methods=['direct','jittered','decorrelated','CPPN','GAN-pool5','GAN-fc6','GAN-fc7','GAN-fc8']
  for optim_method in optimization_methods:
    plot_8_random_classes_figure(optim_method=optim_method,image_folder=image_folder,target_parent_folder=figure_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='plot_cat_vs_dog_figure')
    parser.add_argument(
        "--optimization_methods",nargs='+',
        choices= ['direct','jittered','decorrelated','CPPN','GAN-pool5','GAN-fc6','GAN-fc7','GAN-fc8'],
        default= ['direct','jittered','decorrelated','CPPN','GAN-pool5','GAN-fc6','GAN-fc7','GAN-fc8'])
    parser.add_argument(
        "--image_folder",nargs=1,default="optimization_results")
    parser.add_argument(
        "--figure_folder",nargs=1,default="figures")
    args=parser.parse_args()

    plot_batch(**vars(args))