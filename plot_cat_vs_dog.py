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
from third_party.curlyBrace import curlyBrace

# map ugly model names to nice model names
model_name_dict={'InceptionV3':'Inception-v3',
'Resnet50':'ResNet-50',
'Resnet_50_l2_eps5':'$\ell_2$-adv-trained (${\epsilon=5}$)\nResNet-50\n',
'Wide_Resnet50_2_l2_eps5':'$\ell_2$-adv-trained (${\epsilon=5}$)\nWRN-50-2'}

def plot_im_matrix(im_matrix,x_class,y_class,c,subplot_spec=None,cmap_matrix=None,upscale=None,exp_type='imagenet'):

    n_models=len(models_to_plot)
    # c is a configuration dict
    if subplot_spec is None:
        #   start figure
        figsize=[np.sum([c.left_margin,c.inch_per_image*n_models,c.right_margin]),
                  np.sum([c.top_margin,c.inch_per_image*n_models,c.bottom_margin])]
        print('figure size=',figsize,'inches')
        fig=plt.figure(figsize=figsize)

        gs0 = gridspec.GridSpec(nrows=3, ncols=3,
                        height_ratios=[c.top_margin,c.inch_per_image*n_models,c.bottom_margin],
                        width_ratios=[c.left_margin,c.inch_per_image*n_models,c.right_margin],
                        figure=fig,wspace=0,hspace=0,left=0,right=1,bottom=0,top=1)
    else:
        fig=plt.gcf()
        gs0 = gridspec.GridSpecFromSubplotSpec(nrows=3, ncols=3,
                        height_ratios=[c.top_margin,c.inch_per_image*n_models,c.bottom_margin],
                        width_ratios=[c.left_margin,c.inch_per_image*n_models,c.right_margin],
                        subplot_spec=subplot_spec,wspace=0,hspace=0)

    if c.do_plot_images:
        gs00 = gridspec.GridSpecFromSubplotSpec(nrows=n_models, ncols=n_models, subplot_spec=gs0[1,1],hspace=0.0,wspace=0.0)
        for i_row,row_model_name in enumerate(models_to_plot):
           for i_col,col_model_name in enumerate(models_to_plot):
              cur_im=im_matrix[i_row, i_col]
              if cmap_matrix is not None:
                  cmap=cmap_matrix[i_row, i_col]
              else:
                  cmap='gray'
              ax = plt.subplot(gs00[i_row, i_col])
              if type(cur_im) is np.ndarray or isinstance(cur_im,PIL.Image.Image):
                  if exp_type=='MNIST':
                      ax.imshow(1.0-cur_im,cmap=cmap, interpolation='nearest',extent=[0,1,0,1])
                  elif exp_type in ['CIFAR-10','imagenet']:
                      if upscale is not None:
                          cur_im = cur_im.resize((upscale,upscale), resample=PIL.Image.LANCZOS)
                      ax.imshow(cur_im,extent=[0,1,0,1])
                  else:
                      raise NotImplementedError

              elif i_row==i_col: # diagonal line

                  ax.imshow(np.ones([1,1,3]),extent=[0,1,0,1])
                  ax.plot([1, 0], [0, 1], 'k-',linewidth=c.im_matrix_line_width,transform=ax.transAxes)
              elif cur_im == 'blank':
                  ax.imshow(np.ones([1,1,3]),extent=[0,1,0,1])
                  ax.plot([1, 0], [0, 1], 'k-',linewidth=c.im_matrix_line_width,transform=ax.transAxes)
              else: # a missing plot
                  ax.imshow(np.asarray([1.0,0.5,0.5]).reshape([1,1,3]),extent=[0,1,0,1])
                  plt.xlim([0,1])
                  plt.ylim([0,1])

              for spine in ax.spines.values():
                  spine.set_linewidth(c.spine_line_width)
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
                   labelleft=False) # labels along the bottom edge are off

              if i_row==0: # is it the top row?
                 if not (hasattr(c,'omit_top_model_labels') and c.omit_top_model_labels):
                     ax.text(0.25,1.05,model_name_to_title(col_model_name),ha='left',clip_on=False,va='bottom',rotation=45,fontdict={'fontsize':c.model_name_font_size})

              if i_col==0: # is it the leftmost column?
                  if not (hasattr(c,'omit_left_model_labels') and c.omit_left_model_labels):
                      ax.text(-0.15,0.5,model_name_to_title(row_model_name),ha='right',clip_on=False,va='center',fontsize=c.model_name_font_size)

              plt.setp(ax.spines.values(), color='black')

              bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
              width, height = bbox.width, bbox.height
              dpi=28/0.04676587926509179

    if c.do_plot_horizontal_brace:
        x_major_label_ax=plt.Subplot(fig,gs0[0,1])
        fig.add_subplot(x_major_label_ax)
        x_major_label_ax.set_axis_off()
        plt.ylim([0,1])
        plt.xlim([0,n_models])
        curlyBrace(fig=plt.gcf(), ax=x_major_label_ax,
                   p2=[n_models-0.25+c.top_right_curly_brace_horizontal_shift,c.h_curly_pad],
                   p1=[0.25+c.top_left_curly_brace_horizontal_shift,c.h_curly_pad],
                   k_r=0.05, bool_auto=True,
                   str_text='models targeted\n to recognize '+r"$\bf{" + str(x_class.replace('_','\>'))+"}$", int_line_num=3,color='black',fontdict={'fontsize':c.major_label_font_size},linewidth=c.curly_brace_line_width,clip_on=False,)

    if c.do_plot_vertical_brace:
        y_major_label_ax=plt.Subplot(fig,gs0[1,0])
        fig.add_subplot(y_major_label_ax)
        y_major_label_ax.set_axis_off()
        plt.ylim([0,n_models])
        plt.xlim([0,1])
      #  y_major_label_ax.text(0.25,0.5,'model targeted\n to recognize {}'.format(y_class),fontdict={'fontsize':c.major_label_font_size},verticalalignment='center',horizontalalignment='left',rotation=90)
        curlyBrace(fig=plt.gcf(), ax=y_major_label_ax, p2=[c.v_curly_pad,n_models-0.5], p1=[c.v_curly_pad,0.5], k_r=0.05, bool_auto=True,
                   str_text='models targeted\n to recognize '+r"$\bf{" + str(y_class.replace('_','\>'))+"}$", int_line_num=3,color='black',fontdict={'fontsize':c.major_label_font_size},linewidth=c.curly_brace_line_width,clip_on=False,nudge_label_x=c.nudge_v_label_x, nudge_label_y=c.nudge_v_label_y)


def model_name_to_title(model_name):
    if model_name in model_name_dict.keys():
        return model_name_dict[model_name]
    else:
        return model_name.replace('_',' ')

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
    # make sure the folder doesn't have mixed files (the first model is always one model, and the second is always the other)
    assert len(png_files_info.model_1_name.unique())==1 and len(png_files_info.model_2_name.unique())==1
    model_1_name=png_files_info.model_1_name.unique()[0]
    model_2_name=png_files_info.model_2_name.unique()[0]
    assert ((model_1_name == rows_model) and (model_2_name == columns_model) or
            (model_1_name == columns_model) and (model_2_name == rows_model))

    class_names=np.unique(list(png_files_info.model_1_target)+list(png_files_info.model_2_target))
    n_classes=len(class_names)
    im_matrix=np.empty((n_classes,n_classes), dtype=np.object)

    for png_file,model_1_target,model_2_target in tqdm(zip(png_files_info.filename,png_files_info.model_1_target,png_files_info.model_2_target)):
        #check response statistics - is it a successful crafting?
        cur_im=plt.imread(png_file)

        model_1_target_idx=class_names.index(model_1_target)
        model_2_target_idx=class_names.index(model_2_target)

        if model_1_name==rows_model and model_2_name==columns_model:
            im_matrix[model_1_target_idx, model_2_target_idx]=cur_im
        else:
            im_matrix[model_2_target_idx, model_1_target_idx]=cur_im

    plot_im_matrix(im_matrix,rows_model,columns_model,c,subplot_spec=subplot_spec,panel_label=panel_label)

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


def plot_single_class_pair_multiple_model_pairs_controversial_stimuli_matrix(x_class,y_class,c,stimuli_path=None,subplot_spec=None,png_files_info=None,upscale=None):
    print(models_to_plot)
    if png_files_info is None:
        n_models=len(models_to_plot)
        png_files=glob.glob(os.path.join(stimuli_path,'**/*.png'))+glob.glob(os.path.join(stimuli_path,'*.png'))
        assert len(png_files)>0, "no png files found in "+stimuli_path
        print("found {} png files.".format(len(png_files)))
        png_files_info=get_png_file_info(png_files)

    # filter pngs to match required classes
    cur_subplot_files_mask=np.logical_and(
                np.logical_or(
                        np.logical_and(png_files_info.model_1_target==x_class,png_files_info.model_2_target==y_class),
                        np.logical_and(png_files_info.model_2_target==x_class,png_files_info.model_1_target==y_class)),
                np.logical_and(
                        [model_name in models_to_plot for model_name in png_files_info.model_1_name],
                        [model_name in models_to_plot for model_name in png_files_info.model_2_name]
                        )
                )

    selected_png_files=png_files_info[cur_subplot_files_mask]

    im_matrix=np.empty((n_models,n_models), dtype=np.object)

    for f in selected_png_files.itertuples():

        if f.model_1_target==x_class and f.model_2_target==y_class:
            col=models_to_plot.index(f.model_1_name)
            row=models_to_plot.index(f.model_2_name)
        else:
            col=models_to_plot.index(f.model_2_name)
            row=models_to_plot.index(f.model_1_name)

        cur_im=PIL.Image.open(f.filename)

        im_matrix[row, col]=cur_im
    plot_im_matrix(im_matrix,x_class,y_class,c,subplot_spec=subplot_spec,upscale=upscale)


def plot_cat_vs_dog_figure(optim_method='direct',target_parent_folder='figures',image_folder='optimization_results'):
    global models_to_plot
    models_to_plot=['InceptionV3','Resnet50','Resnet_50_l2_eps5','Wide_Resnet50_2_l2_eps5']

    x_class='Persian_cat'
    y_class='Weimaraner'

    # figure configuration
    c=AttrDict()
    c.top_margin=1.5
    c.bottom_margin=0.025
    c.left_margin=1.5
    c.right_margin=0.33

    c.model_name_font_size=6
    c.major_label_font_size=8
    c.inch_per_minor_title_space=1.4
    c.inch_per_image=(3.42-c.left_margin-c.right_margin)/3.42 # 5.0 inch is the total with of the figure. There are 4 models.
    #between_subplot_margin=16/72
    c.curly_brace_line_width=0.5
    c.nudge_h_curly_label=0
    c.nudge_v_curly_label=0

    c.top_left_curly_brace_horizontal_shift=2.02-0.25-1.0
    c.top_right_curly_brace_horizontal_shift=1.09+0.25-0.5
    c.im_matrix_line_width=0.75
    c.spine_line_width=0.8
    c.h_curly_pad=0.55
    c.v_curly_pad=0.32

    c.nudge_v_label_x=0
    c.nudge_v_label_y=0

    c.do_plot_images=True
    #major_margin=16/72
    #inch_per_major_title_space=32/72

    c.do_plot_vertical_brace=True
    c.do_plot_horizontal_brace=True

    upscale=256
    stimuli_path=os.path.join(image_folder,optim_method+'_optim_cat_vs_dog')

    plt.close('all')
    plot_single_class_pair_multiple_model_pairs_controversial_stimuli_matrix(x_class=x_class,y_class=y_class,c=c,stimuli_path=stimuli_path,upscale=upscale)

    fig_fname=optim_method+'_optim_'+str(len(models_to_plot))+"_models_"+str(x_class)+"_by_" + str(y_class) + ".pdf"
    pathlib.Path(target_parent_folder).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(target_parent_folder,fig_fname),dpi=upscale/c.inch_per_image)
    plt.savefig(os.path.join(target_parent_folder,fig_fname.replace('.pdf','.png')),dpi=upscale/c.inch_per_image)

    print('saved to',os.path.join(target_parent_folder,fig_fname))


def plot_batch(optimization_methods=None,image_folder='optimization_results',figure_folder='figures'):
  if optimization_methods is None:
    optimization_methods=['direct','jittered','decorrelated','CPPN','GAN-pool5','GAN-fc6','GAN-fc7','GAN-fc8']
  for optim_method in optimization_methods:
    plot_cat_vs_dog_figure(optim_method=optim_method,image_folder=image_folder,target_parent_folder=figure_folder)

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